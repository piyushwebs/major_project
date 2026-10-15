import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://localhost:5000";

function App() {
  const [eventsText, setEventsText] = useState(
    "E5, E22, E11, E9, E11, E9, E26",
  );
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [logFile, setLogFile] = useState(null);
  const [parseResult, setParseResult] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("Backend unavailable");
        return response.json();
      })
      .then((data) => {
        setBackendStatus(
          data.modelConnected ? "Model connected" : "Model disconnected",
        );
      })
      .catch(() => setBackendStatus("Disconnected"));
  }, []);

  async function handleAnalyze(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    const events = eventsText
      .split(/[\s,]+/)
      .map((item) => item.trim().toUpperCase())
      .filter(Boolean);

    if (events.length === 0) {
      setError("Enter at least one event ID.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Could not process this request.");
      }

      setResult(data);
    } catch (err) {
      setError(
        `${err.message}. Check that the Express backend is running on port 5000.`,
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(event) {
    event.preventDefault();

    setUploadError("");
    setParseResult(null);

    if (!logFile) {
      setUploadError("Please select a .log or .txt file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", logFile);

    setUploading(true);

    try {
      const response = await fetch(`${API_URL}/api/parse-log`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || data.detail || "The file could not be parsed.",
        );
      }

      setParseResult(data);
    } catch (err) {
      setUploadError(
        `${err.message}. Check that Express and FastAPI are running.`,
      );
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">⌁</div>
          <div>
            <h2>LogGuard</h2>
            <span>ANOMALY DETECTION</span>
          </div>
        </div>

        <nav className="navigation">
          <div className="nav-item active">▦ &nbsp; Dashboard</div>
          <div className="nav-item">⌕ &nbsp; Log Analysis</div>
          <div className="nav-item">◷ &nbsp; Prediction History</div>
          <div className="nav-item">⚙ &nbsp; Settings</div>
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          Local development
          <p>HDFS · LSTM model</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">MONITORING / OVERVIEW</p>
            <h1>Log Analysis Dashboard</h1>
          </div>
          <div className="connection">
            <span
              className={`status-dot ${
                backendStatus === "Model connected" ? "" : "offline"
              }`}
            />
            Backend: {backendStatus}
          </div>
        </header>

        <section className="welcome">
          <div>
            <p className="eyebrow">SYSTEM INTELLIGENCE</p>
            <h2>Detect unusual behaviour in system logs.</h2>
            <p>
              Submit an event sequence to prepare an anomaly analysis request.
            </p>
          </div>
          <div className="welcome-symbol">⌁</div>
        </section>

        <section className="stats-grid">
          <article className="stat-card">
            <span>Detection model</span>
            <h3>LSTM</h3>
            <p>Sequence-based classification</p>
          </article>
          <article className="stat-card">
            <span>Accepted event IDs</span>
            <h3>E1 – E29</h3>
            <p>HDFS event vocabulary</p>
          </article>
          <article className="stat-card">
            <span>Maximum sequence length</span>
            <h3>50 events</h3>
            <p>Matches model preprocessing</p>
          </article>
        </section>

        <section className="workspace">
          <article className="panel input-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">INPUT</p>
                <h2>Analyze event sequence</h2>
              </div>
              <span className="step-badge">01</span>
            </div>

            <p className="description">
              Enter event IDs separated by commas or spaces. For now, the API
              accepts parsed event IDs rather than raw log lines.
            </p>

            <form onSubmit={handleAnalyze}>
              <label htmlFor="events">Event sequence</label>
              <textarea
                id="events"
                value={eventsText}
                onChange={(event) => setEventsText(event.target.value)}
                placeholder="E5, E22, E11, E9..."
                rows={5}
              />

              <div className="input-hint">Example: E5, E22, E11, E9, E26</div>

              {error && <p className="error-message">{error}</p>}

              <button type="submit" disabled={loading}>
                {loading ? "Sending request..." : "Submit for analysis →"}
              </button>
            </form>
          </article>

          <article className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">OUTPUT</p>
                <h2>Analysis response</h2>
              </div>
              <span className="step-badge">02</span>
            </div>

            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">⌕</div>
                <h3>Waiting for input</h3>
                <p>
                  Submit an event sequence to see the backend response here.
                </p>
              </div>
            ) : (
              <div className="result-content">
                {result.mock && (
                  <div className="mock-warning">
                    TEST RESPONSE — MODEL NOT CONNECTED
                  </div>
                )}

                {!result.mock && (
                  <div className="live-warning">
                    ✓ REAL PREDICTION — TRAINED LSTM MODEL
                  </div>
                )}

                <div
                  className={`prediction-label ${
                    result.predicted_label === "Anomaly" ? "anomaly" : "normal"
                  }`}
                >
                  {result.predicted_label || "Request received"}
                </div>

                <p className="result-message">{result.message}</p>

                {result.predicted_label && (
                  <div className="probability-section">
                    <div className="result-row">
                      <span>Anomaly probability</span>
                      <strong>
                        {(Number(result.anomaly_probability) * 100).toFixed(4)}%
                      </strong>
                    </div>

                    <div className="probability-track">
                      <div
                        className={`probability-fill ${
                          result.predicted_label === "Anomaly" ? "anomaly" : ""
                        }`}
                        style={{
                          width: `${Number(result.anomaly_probability) * 100}%`,
                        }}
                      />
                    </div>

                    <p className="input-hint">
                      Classification threshold:{" "}
                      {(Number(result.threshold) * 100).toFixed(0)}%
                    </p>
                  </div>
                )}

                <div className="result-row">
                  <span>Events received</span>
                  <strong>{result.eventsReceived}</strong>
                </div>

                <div className="result-row">
                  <span>Events used</span>
                  <strong>{result.events_used}</strong>
                </div>

                <div className="result-row">
                  <span>Model status</span>
                  <strong className="success-text">
                    {result.mock ? "Mock" : "Connected"}
                  </strong>
                </div>

                <label>Analyzed event sequence</label>

                <div className="event-list">
                  {result.events?.map((item, index) => (
                    <span className="event-chip" key={`${item}-${index}`}>
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </article>
        </section>

        <section className="upload-workspace">
          <article className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">RAW LOG INGESTION</p>
                <h2>Upload and parse log file</h2>
              </div>
              <span className="step-badge">03</span>
            </div>

            <p className="description">
              Upload an HDFS .log or .txt file. LogGuard will match log messages
              against the HDFS templates and summarize the event sequences it
              can reconstruct.
            </p>

            <form onSubmit={handleUpload}>
              <label htmlFor="log-file">Choose log file</label>

              <input
                id="log-file"
                className="file-input"
                type="file"
                accept=".log,.txt,text/plain"
                onChange={(event) => {
                  setLogFile(event.target.files?.[0] || null);
                  setParseResult(null);
                  setUploadError("");
                }}
              />

              {logFile && (
                <p className="input-hint">
                  Selected: {logFile.name} · {(logFile.size / 1024).toFixed(1)}{" "}
                  KB
                </p>
              )}

              <p className="input-hint">Maximum file size: 20 MB</p>

              {uploadError && <p className="error-message">{uploadError}</p>}

              <button type="submit" disabled={uploading || !logFile}>
                {uploading ? "Parsing log file..." : "Upload and parse →"}
              </button>
            </form>
          </article>

          {parseResult && (
            <article className="panel parse-results">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">PARSER REPORT</p>
                  <h2>Parsing results</h2>
                </div>
                <span className="step-badge">04</span>
              </div>

              <p className="description">File: {parseResult.filename}</p>

              <div className="parse-stats">
                <div className="parse-stat">
                  <span>Total lines</span>
                  <strong>{parseResult.lines_total}</strong>
                </div>

                <div className="parse-stat">
                  <span>Matched</span>
                  <strong>{parseResult.matched_lines}</strong>
                </div>

                <div className="parse-stat">
                  <span>Unmatched</span>
                  <strong>{parseResult.unmatched_lines}</strong>
                </div>

                <div className="parse-stat">
                  <span>Malformed</span>
                  <strong>{parseResult.malformed_lines}</strong>
                </div>

                <div className="parse-stat">
                  <span>Parser coverage</span>
                  <strong>{parseResult.parser_coverage_percent}%</strong>
                </div>

                <div className="parse-stat">
                  <span>Unique blocks</span>
                  <strong>{parseResult.unique_blocks}</strong>
                </div>
              </div>

              <h3 className="subheading">Model vocabulary compatibility</h3>

              <div className="result-row">
                <span>Compatible traces</span>
                <strong>{parseResult.model_compatible_trace_count}</strong>
              </div>

              <div className="result-row">
                <span>Incompatible traces</span>
                <strong>{parseResult.model_incompatible_trace_count}</strong>
              </div>

              {Object.keys(parseResult.unsupported_event_counts || {}).length >
                0 && (
                <>
                  <h3 className="subheading">Unsupported event IDs</h3>
                  <div className="event-list">
                    {Object.entries(parseResult.unsupported_event_counts).map(
                      ([eventId, count]) => (
                        <span className="event-chip warning-chip" key={eventId}>
                          {eventId}: {count}
                        </span>
                      ),
                    )}
                  </div>
                </>
              )}

              <h3 className="subheading">Reconstructed trace preview</h3>

              {parseResult.traces_preview?.length > 0 ? (
                <div className="trace-list">
                  {parseResult.traces_preview.map((trace) => (
                    <div className="trace-card" key={trace.block_id}>
                      <div className="trace-heading">
                        <strong>{trace.block_id}</strong>
                        <span>{trace.event_count} events</span>
                      </div>

                      <div className="event-list">
                        {trace.events_preview.map((eventId, index) => (
                          <span
                            className="event-chip"
                            key={`${trace.block_id}-${index}`}
                          >
                            {eventId}
                          </span>
                        ))}
                      </div>

                      <p className="input-hint">
                        {trace.model_compatible
                          ? "Uses supported event IDs"
                          : `Unsupported: ${trace.unsupported_event_ids.join(", ")}`}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="description">
                  No block-level traces could be reconstructed from this file.
                </p>
              )}

              {parseResult.unmatched_samples?.length > 0 && (
                <>
                  <h3 className="subheading">Unmatched line samples</h3>

                  {parseResult.unmatched_samples.map((sample, index) => (
                    <div
                      className="unmatched-item"
                      key={`${sample.line_number}-${index}`}
                    >
                      <strong>
                        Line {sample.line_number} — {sample.status}
                      </strong>
                      <p>{sample.line}</p>
                    </div>
                  ))}
                </>
              )}

              <p className="input-hint">
                Parsing only: no LSTM prediction has been run on this upload
                yet.
              </p>
            </article>
          )}
        </section>

        <footer>
          LogGuard · HDFS Log Anomaly Detection Project
          <span>Development version</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
