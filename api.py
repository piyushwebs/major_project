
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from log_parser import parse_hdfs_log_text
from inference import predict_events, predict_many_events, DEVICE


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LogGuard ML API",
    description="HDFS Log Anomaly Detection using LSTM",
    version="1.0.0",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
DEFAULT_THRESHOLD = 0.5


# --------------------------------------------------
# Request schema for direct event prediction
# --------------------------------------------------

class PredictionRequest(BaseModel):
    events: list[str] = Field(min_length=1, max_length=500)
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelConnected": True,
        "device": str(DEVICE),
    }


# --------------------------------------------------
# Predict a manually supplied event sequence
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_events(
            request.events,
            threshold=request.threshold,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception("Direct model prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Model prediction failed. Check the API terminal.",
        )


# --------------------------------------------------
# Upload, parse and predict from a raw HDFS log file
# --------------------------------------------------

@app.post("/parse-log")
async def parse_log(file: UploadFile = File(...)):
    # 1. Validate filename and extension.
    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() not in {".log", ".txt"}:
        await file.close()

        raise HTTPException(
            status_code=400,
            detail="Upload a .log or .txt file.",
        )

    # 2. Read no more than the configured upload limit plus one byte.
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 20 MB upload limit.",
        )

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # 3. Decode the raw log file.
    try:
        raw_text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="The log file must use UTF-8 text encoding.",
        )

    # 4. Parse raw log messages and reconstruct block traces.
    try:
        parsed = parse_hdfs_log_text(raw_text)

    except FileNotFoundError:
        logger.exception("HDFS template file is missing")

        raise HTTPException(
            status_code=500,
            detail="HDFS event templates could not be found.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception("HDFS log parsing failed")

        raise HTTPException(
            status_code=500,
            detail="Log parsing failed. Check the API terminal.",
        )

    # 5. Select non-empty traces containing only supported event IDs.
    compatible_traces = [
        trace
        for trace in parsed["traces"]
        if trace["model_compatible"] and trace["events"]
    ]

    # 6. Run the existing trained LSTM on compatible traces.
    try:
        predictions = predict_many_events(
            [trace["events"] for trace in compatible_traces],
            threshold=DEFAULT_THRESHOLD,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        logger.exception("Batch LSTM prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed. Check the API terminal.",
        )

    # 7. Associate each prediction with its block ID.
    predictions_by_block = {}

    prediction_counts = {
        "Normal": 0,
        "Anomaly": 0,
    }

    for trace, prediction in zip(compatible_traces, predictions):
        predictions_by_block[trace["block_id"]] = prediction

        prediction_counts[prediction["predicted_label"]] += 1

    # 8. Build a limited preview for the frontend.
    # Keep the full trace collection out of the HTTP response.
    trace_prediction_preview = []

    for trace in parsed["traces"][:10]:
        item = {
            "block_id": trace["block_id"],
            "event_count": trace["event_count"],
            "events_preview": trace["events"][:50],
            "model_compatible": trace["model_compatible"],
            "unsupported_event_ids": trace["unsupported_event_ids"],
        }

        prediction = predictions_by_block.get(trace["block_id"])

        if prediction is not None:
            item.update(prediction)

        trace_prediction_preview.append(item)

    # 9. Return parser statistics and actual model predictions.
    return {
        "success": True,
        "filename": filename,
        "bytes_received": len(content),

        "parser": parsed["parser"],
        "lines_total": parsed["lines_total"],
        "non_empty_lines": parsed["non_empty_lines"],
        "matched_lines": parsed["matched_lines"],
        "unmatched_lines": parsed["unmatched_lines"],
        "malformed_lines": parsed["malformed_lines"],
        "matched_without_block_id": parsed[
            "matched_without_block_id"
        ],
        "parser_coverage_percent": parsed[
            "parser_coverage_percent"
        ],

        "unique_blocks": parsed["unique_blocks"],
        "unique_event_types": parsed["unique_event_types"],
        "event_counts": parsed["event_counts"],

        "unsupported_event_counts": parsed[
            "unsupported_event_counts"
        ],

        "model_compatible_trace_count": parsed[
            "model_compatible_trace_count"
        ],
        "model_incompatible_trace_count": parsed[
            "model_incompatible_trace_count"
        ],

        "predicted_trace_count": len(predictions),
        "prediction_counts": prediction_counts,
        "threshold": DEFAULT_THRESHOLD,

        "traces_preview": trace_prediction_preview,
        "unmatched_samples": parsed["unmatched_samples"],
    }
