const multer = require("multer");
const path = require("path");
const FormData = require("form-data");

const express = require("express");
const cors = require("cors");
const axios = require("axios");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 5000;
const ML_API_URL = "http://127.0.0.1:8000";

app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),

  limits: {
    fileSize: 20 * 1024 * 1024, // 20 MB
  },

  fileFilter: (req, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();

    if (![".log", ".txt"].includes(extension)) {
      return callback(new Error("Only .log and .txt files are allowed."));
    }

    callback(null, true);
  },
});

// Check Express and Python model service status
app.get("/api/health", async (req, res) => {
  let modelConnected = false;

  try {
    const response = await axios.get(`${ML_API_URL}/health`, {
      timeout: 3000,
    });

    modelConnected = response.data.modelConnected === true;
  } catch {
    modelConnected = false;
  }

  res.json({
    status: "ok",
    message: "Log Anomaly Detection backend is running",
    modelConnected,
  });
});

// Send event sequences to the actual trained LSTM model
app.post("/api/predict", async (req, res) => {
  const { events, threshold = 0.5 } = req.body;

  if (
    !Array.isArray(events) ||
    events.length === 0 ||
    events.length > 500
  ) {
    return res.status(400).json({
      error: "Provide between 1 and 500 event IDs.",
    });
  }

  const normalizedEvents = events.map((event) =>
    typeof event === "string" ? event.trim().toUpperCase() : ""
  );

  const validEvents = normalizedEvents.every((event) =>
    /^E([1-9]|1\d|2\d)$/.test(event)
  );

  if (!validEvents) {
    return res.status(400).json({
      error: "Each event must be a valid event ID from E1 to E29.",
    });
  }

  if (
    typeof threshold !== "number" ||
    !Number.isFinite(threshold) ||
    threshold < 0 ||
    threshold > 1
  ) {
    return res.status(400).json({
      error: "Threshold must be a number between 0 and 1.",
    });
  }

  try {
    const response = await axios.post(
      `${ML_API_URL}/predict`,
      {
        events: normalizedEvents,
        threshold,
      },
      { timeout: 15000 }
    );

    const prediction = response.data;

    // Return model results to the React frontend
    return res.json({
      message: "Prediction completed by the trained LSTM model",
      modelConnected: true,
      mock: false,
      predicted_label: prediction.predicted_label,
      anomaly_probability: prediction.anomaly_probability,
      threshold: prediction.threshold,
      eventsReceived: prediction.events_received,
      events_used: prediction.events_used,
      events: normalizedEvents,
    });
  } catch (error) {
    if (error.response) {
      return res.status(
        error.response.status >= 400 && error.response.status < 500
          ? error.response.status
          : 502
      ).json({
        error:
          error.response.data?.detail ||
          "The Python model service could not process the request.",
      });
    }

    return res.status(503).json({
      error:
        "Python model service is unavailable. Check that FastAPI is running on port 8000.",
    });
  }
});

app.post(
  "/api/parse-log",
  (req, res, next) => {
    upload.single("file")(req, res, (err) => {
      if (err) {
        const status = err.code === "LIMIT_FILE_SIZE" ? 413 : 400;

        return res.status(status).json({
          error: err.message,
        });
      }

      next();
    });
  },

  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        error: "Please upload a .log or .txt file.",
      });
    }

    const form = new FormData();

    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: "text/plain",
    });

    try {
      const response = await axios.post(
        `${ML_API_URL}/parse-log`,
        form,
        {
          headers: form.getHeaders(),
          timeout: 120000,
          maxBodyLength: Infinity,
          maxContentLength: Infinity,
        }
      );

      return res.json({
        success: true,
        ...response.data,
      });
    } catch (error) {
      if (error.response) {
        return res.status(
          error.response.status >= 400 &&
          error.response.status < 500
            ? error.response.status
            : 502
        ).json({
          error:
            error.response.data?.detail ||
            "The Python parser could not process the uploaded file.",
        });
      }

      return res.status(503).json({
        error:
          "Cannot reach the Python parsing service. Check that FastAPI is running on port 8000.",
      });
    }
  }
);

app.use((req, res) => {
  res.status(404).json({ error: "API route not found" });
});

app.listen(PORT, () => {
  console.log(`Backend running at http://localhost:${PORT}`);
});
