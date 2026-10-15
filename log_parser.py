
import csv
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TEMPLATE_PATHS = [
    BASE_DIR / "HDFS_v1" / "preprocessed" / "HDFS.log_templates.csv",
    BASE_DIR / "HDFS_v1" / "preprocessed" / "HDFS_templates.csv",
    BASE_DIR / "HDFS_v1" / "HDFS_templates.csv",
    BASE_DIR / "HDFS_templates.csv",
]

# The currently trained model accepts E1-E29.
MODEL_SUPPORTED_EVENTS = {f"E{i}" for i in range(1, 30)}

# Example:
# 081109 203615 148 INFO dfs.DataNode$PacketResponder:
# PacketResponder 1 for block blk_123 terminating
HDFS_LINE_PATTERN = re.compile(
    r"^\s*"
    r"(?P<date>\S+)\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<pid>\S+)\s+"
    r"(?P<level>\S+)\s+"
    r"(?P<component>[^:]+):\s*"
    r"(?P<content>.*)$"
)

BLOCK_PATTERN = re.compile(r"\bblk_-?\d+\b", re.IGNORECASE)


def find_template_file():
    """Find the canonical HDFS event-template CSV."""
    for path in TEMPLATE_PATHS:
        if path.is_file():
            return path

    raise FileNotFoundError(
        "HDFS_templates.csv not found. Checked:\n"
        + "\n".join(str(path) for path in TEMPLATE_PATHS)
    )



def compile_template(template):
    """
    Convert HDFS template wildcards into regex patterns.

    Supports both:
      [*]  - wildcard format in your CSV
      <*>  - standard template wildcard format
    """
    normalized = template.replace("<*>", "[*]")
    parts = normalized.split("[*]")

    regex = ".*?".join(
        re.escape(part) for part in parts
    )

    return re.compile(
        "^" + regex + "$",
        flags=re.IGNORECASE,
    )



def load_templates():
    """Load and compile HDFS event templates."""
    template_path = find_template_file()
    templates = []

    with template_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"EventId", "EventTemplate"}

        if not reader.fieldnames:
            raise ValueError("The template CSV has no header.")

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise ValueError(
                f"Template CSV is missing columns: {sorted(missing)}"
            )

        for row in reader:
            event_id = (row.get("EventId") or "").strip().upper()
            template = (row.get("EventTemplate") or "").strip()

            if not event_id or not template:
                continue

            if not re.fullmatch(r"E\d+", event_id):
                continue

            templates.append({
                "event_id": event_id,
                "template": template,
                "regex": compile_template(template),
                "specificity": len(template.replace("<*>", "")),
            })

    if not templates:
        raise ValueError("No valid event templates were loaded.")

    # More specific templates are considered before broader templates.
    templates.sort(
        key=lambda item: item["specificity"],
        reverse=True,
    )

    return templates


def parse_raw_line(raw_line):
    """
    Split an HDFS raw line into its structured header and message content.

    Returns None when the line does not match the expected HDFS format.
    """
    match = HDFS_LINE_PATTERN.fullmatch(raw_line.strip())

    if not match:
        return None

    return match.groupdict()


def match_event(content, templates):
    """
    Match message content against the canonical templates.

    Returns the best first match according to template specificity.
    Returns None if no template matches.
    """
    for item in templates:
        if item["regex"].fullmatch(content):
            return item

    return None


def parse_hdfs_log_text(raw_text):
    """
    Parse raw HDFS log text and create block-level event sequences.

    This performs parsing and grouping only. It does not run the LSTM.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("The uploaded log file is empty.")

    templates = load_templates()
    template_path = find_template_file()

    block_traces = {}
    event_counts = Counter()
    unmatched_samples = []

    total_lines = 0
    non_empty_lines = 0
    matched_lines = 0
    unmatched_lines = 0
    malformed_lines = 0
    matched_without_block_id = 0

    for line_number, raw_line in enumerate(
        raw_text.splitlines(),
        start=1,
    ):
        total_lines += 1
        line = raw_line.strip()

        if not line:
            continue

        non_empty_lines += 1

        # 1. Separate the raw log header from its message.
        parsed_line = parse_raw_line(line)

        if parsed_line is None:
            malformed_lines += 1

            if len(unmatched_samples) < 10:
                unmatched_samples.append({
                    "line_number": line_number,
                    "status": "malformed",
                    "line": line[:250],
                })

            continue

        content = parsed_line["content"]

        # 2. Match only the message content against templates.
        matched_template = match_event(content, templates)

        if matched_template is None:
            unmatched_lines += 1

            if len(unmatched_samples) < 10:
                unmatched_samples.append({
                    "line_number": line_number,
                    "status": "unknown_template",
                    "line": line[:250],
                    "content": content[:250],
                })

            continue

        matched_lines += 1
        event_id = matched_template["event_id"]
        event_counts[event_id] += 1

        # 3. Extract block IDs from the log message.
        block_ids = list(dict.fromkeys(
            item.group(0)
            for item in BLOCK_PATTERN.finditer(content)
        ))

        if not block_ids:
            matched_without_block_id += 1
            continue

        # 4. Append the event to each referenced block's trace.
        for block_id in block_ids:
            if block_id not in block_traces:
                block_traces[block_id] = []

            block_traces[block_id].append({
                "event_id": event_id,
                "line_number": line_number,
                "date": parsed_line["date"],
                "time": parsed_line["time"],
            })

    # 5. Check whether each trace uses only supported event IDs.
    traces = []
    compatible_trace_count = 0
    incompatible_trace_count = 0
    unsupported_event_counts = Counter()

    for block_id, records in block_traces.items():
        events = [record["event_id"] for record in records]

        unsupported = sorted(
            set(events) - MODEL_SUPPORTED_EVENTS
        )

        compatible = len(unsupported) == 0

        if compatible:
            compatible_trace_count += 1
        else:
            incompatible_trace_count += 1

            for event_id in unsupported:
                unsupported_event_counts[event_id] += events.count(
                    event_id
                )

        traces.append({
            "block_id": block_id,
            "events": events,
            "event_count": len(events),
            "model_compatible": compatible,
            "unsupported_event_ids": unsupported,
        })

    coverage = (
        matched_lines / non_empty_lines * 100
        if non_empty_lines
        else 0.0
    )

    traces_preview = []

    for trace in traces[:10]:
        traces_preview.append({
            "block_id": trace["block_id"],
            "event_count": trace["event_count"],
            "events_preview": trace["events"][:50],
            "model_compatible": trace["model_compatible"],
            "unsupported_event_ids": trace["unsupported_event_ids"],
        })

    # Keep all reconstructed traces available for the next pipeline stage.
    # The API should return only the summary and a limited preview to React.
    return {
        "parser": "HDFS template matcher",
        "template_file": str(template_path),
        "lines_total": total_lines,
        "non_empty_lines": non_empty_lines,
        "matched_lines": matched_lines,
        "unmatched_lines": unmatched_lines,
        "malformed_lines": malformed_lines,
        "matched_without_block_id": matched_without_block_id,
        "parser_coverage_percent": round(coverage, 2),
        "unique_blocks": len(traces),
        "unique_event_types": len(event_counts),
        "event_counts": dict(sorted(event_counts.items())),
        "unsupported_event_counts": dict(
            sorted(unsupported_event_counts.items())
        ),
        "model_compatible_trace_count": compatible_trace_count,
        "model_incompatible_trace_count": incompatible_trace_count,
        "traces_preview": traces_preview,
        "unmatched_samples": unmatched_samples,
        "traces": traces,
    }
