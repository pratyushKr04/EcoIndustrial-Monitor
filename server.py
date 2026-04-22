"""
Environmental Monitoring Dashboard — HTTP Server.

Serves the frontend and provides API endpoints:
    GET  /                     → Dashboard (frontend/index.html)
    GET  /api/report           → Compliance report JSON
    GET  /api/maps             → List of map image filenames
    GET  /api/metrics          → Training metrics JSON
    GET  /api/config           → Current system configuration
    GET  /api/status           → Current analysis job status
    POST /api/analyze          → Start analysis for a city
    GET  /outputs/maps/<file>  → Static map images
    GET  /outputs/metrics/<f>  → Static metric images

Usage:
    python server.py              # Starts on port 8000
    python server.py --port 3000  # Custom port
"""

import http.server
import io
import json
import os
import sys
import subprocess
import threading
import time
import argparse
import mimetypes
from urllib.parse import urlparse, unquote

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Force legacy Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Directories
FRONTEND_DIR = os.path.join(ROOT, "frontend")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
REPORT_PATH = os.path.join(OUTPUTS_DIR, "reports", "compliance_report.json")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
MAPS_DIR = os.path.join(OUTPUTS_DIR, "maps")

# ── Background Job State ──────────────────────────────────
job_lock = threading.Lock()
current_job = {
    "status": "idle",       # idle | running | done | error
    "city": None,
    "progress": [],         # list of log lines
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def reset_job():
    """Reset the job state."""
    current_job.update({
        "status": "idle",
        "city": None,
        "progress": [],
        "started_at": None,
        "finished_at": None,
        "error": None,
    })


def run_analysis_background(city_name: str, dates: dict = None):
    """Run inference for a city in a background thread."""
    with job_lock:
        current_job["status"] = "running"
        current_job["city"] = city_name
        current_job["progress"] = [f"Starting analysis for {city_name}..."]
        current_job["started_at"] = time.time()
        current_job["finished_at"] = None
        current_job["error"] = None

    try:
        # Build the inference command
        script = os.path.join(ROOT, "main.py")
        cmd = [sys.executable, script, "--mode", "infer", "--roi", city_name]

        # Add date args if provided
        if dates:
            if dates.get("t1_start"):
                cmd += ["--t1-start", dates["t1_start"]]
            if dates.get("t1_end"):
                cmd += ["--t1-end", dates["t1_end"]]
            if dates.get("t2_start"):
                cmd += ["--t2-start", dates["t2_start"]]
            if dates.get("t2_end"):
                cmd += ["--t2-end", dates["t2_end"]]

        current_job["progress"].append(f"Running: {' '.join(cmd)}")

        # Run as subprocess and capture output line-by-line
        # Pass PYTHONIOENCODING so subprocess prints UTF-8 too
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            encoding='utf-8',
            errors='replace',
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                with job_lock:
                    current_job["progress"].append(line)
                    # Keep only last 100 lines to avoid memory bloat
                    if len(current_job["progress"]) > 100:
                        current_job["progress"] = current_job["progress"][-100:]

        process.wait()

        with job_lock:
            current_job["finished_at"] = time.time()
            if process.returncode == 0:
                current_job["status"] = "done"
                current_job["progress"].append("✅ Analysis complete!")
            else:
                current_job["status"] = "error"
                current_job["error"] = f"Process exited with code {process.returncode}"
                current_job["progress"].append(
                    f"❌ Analysis failed (exit code {process.returncode})")

    except Exception as e:
        with job_lock:
            current_job["status"] = "error"
            current_job["error"] = str(e)
            current_job["finished_at"] = time.time()
            current_job["progress"].append(f"❌ Error: {e}")


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Handles HTTP requests for the dashboard."""

    def do_GET(self):
        path = unquote(urlparse(self.path).path)

        # ── API Routes ──
        if path == "/api/report":
            self._serve_json_file(REPORT_PATH)
        elif path == "/api/maps":
            self._serve_map_list()
        elif path == "/api/metrics":
            self._serve_metrics()
        elif path == "/api/config":
            self._serve_config()
        elif path == "/api/status":
            self._serve_status()

        # ── Static files: outputs ──
        elif path.startswith("/outputs/"):
            rel = path[len("/outputs/"):]
            filepath = os.path.join(OUTPUTS_DIR, rel.replace("/", os.sep))
            self._serve_static(filepath)

        # ── Frontend files ──
        elif path == "/" or path == "/index.html":
            self._serve_static(os.path.join(FRONTEND_DIR, "index.html"))
        else:
            filepath = os.path.join(FRONTEND_DIR, path.lstrip("/").replace("/", os.sep))
            if os.path.isfile(filepath):
                self._serve_static(filepath)
            else:
                self._send_error(404, f"Not found: {path}")

    def do_POST(self):
        path = unquote(urlparse(self.path).path)

        if path == "/api/analyze":
            self._handle_analyze()
        else:
            self._send_error(404, f"Not found: {path}")

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── Analysis endpoint ────────────────────────────────

    def _handle_analyze(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            city = data.get("city", "").strip()
        except Exception:
            self._send_error(400, "Invalid request body. Expected JSON with 'city' field.")
            return

        if not city:
            self._send_error(400, "City name is required.")
            return

        # Extract optional date fields
        dates = {
            "t1_start": data.get("t1_start", ""),
            "t1_end":   data.get("t1_end", ""),
            "t2_start": data.get("t2_start", ""),
            "t2_end":   data.get("t2_end", ""),
        }

        with job_lock:
            if current_job["status"] == "running":
                self._send_json(json.dumps({
                    "status": "already_running",
                    "city": current_job["city"],
                    "message": f"Analysis already running for {current_job['city']}. Wait for it to finish.",
                }))
                return

        # Start background analysis
        thread = threading.Thread(
            target=run_analysis_background, args=(city, dates), daemon=True
        )
        thread.start()

        self._send_json(json.dumps({
            "status": "started",
            "city": city,
            "message": f"Analysis started for {city}",
        }))

    # ── Status endpoint ──────────────────────────────────

    def _serve_status(self):
        with job_lock:
            status_data = {
                "status": current_job["status"],
                "city": current_job["city"],
                "progress": current_job["progress"][-20:],  # Last 20 lines
                "error": current_job["error"],
            }
            if current_job["started_at"]:
                elapsed = (current_job["finished_at"] or time.time()) - current_job["started_at"]
                status_data["elapsed_seconds"] = round(elapsed, 1)
        self._send_json(json.dumps(status_data))

    # ── Existing API Handlers ────────────────────────────

    def _serve_json_file(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = f.read()
            self._send_json(data)
        else:
            self._send_json("[]")

    def _serve_map_list(self):
        if os.path.isdir(MAPS_DIR):
            maps = sorted([
                f for f in os.listdir(MAPS_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
        else:
            maps = []
        self._send_json(json.dumps(maps))

    def _serve_metrics(self):
        metrics = {}
        if os.path.isdir(METRICS_DIR):
            for fname in os.listdir(METRICS_DIR):
                if fname.endswith("_history.json"):
                    key = fname.replace("_history.json", "")
                    fpath = os.path.join(METRICS_DIR, fname)
                    with open(fpath, "r") as f:
                        metrics[key] = json.load(f)
        self._send_json(json.dumps(metrics))

    def _serve_config(self):
        try:
            import config
            cfg = {
                "roi": config.ROI_CONFIG.get("value", "Unknown"),
                "roi_type": config.ROI_CONFIG.get("type", "place"),
                "t1_start": config.T1_START,
                "t1_end": config.T1_END,
                "t2_start": config.T2_START,
                "t2_end": config.T2_END,
                "training_cities": config.TRAINING_CITIES,
                "patch_size": config.PATCH_SIZE,
                "batch_size": config.BATCH_SIZE,
                "epochs": config.EPOCHS,
                "veg_threshold": config.VEG_VIOLATION_THRESHOLD,
            }
        except Exception as e:
            cfg = {"error": str(e)}
        self._send_json(json.dumps(cfg))

    # ── Static File Server ────────────────────────────────

    def _serve_static(self, filepath):
        filepath = os.path.normpath(filepath)

        if not filepath.startswith(ROOT):
            self._send_error(403, "Forbidden")
            return

        if not os.path.isfile(filepath):
            self._send_error(404, f"Not found: {os.path.basename(filepath)}")
            return

        content_type, _ = mimetypes.guess_type(filepath)
        if content_type is None:
            content_type = "application/octet-stream"

        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    # ── Response helpers ──────────────────────────────────

    def _send_json(self, data):
        content = data.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, code, message):
        content = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        print(f"[SERVER] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Environmental Monitoring Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    args = parser.parse_args()

    # Use ThreadingHTTPServer for concurrent requests during analysis
    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)

    print("=" * 60)
    print("  ENVIRONMENTAL MONITORING DASHBOARD")
    print("=" * 60)
    print(f"  🌐 Dashboard: http://localhost:{args.port}")
    print(f"  📁 Frontend:  {FRONTEND_DIR}")
    print(f"  📊 Outputs:   {OUTPUTS_DIR}")
    print(f"  ⏹  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
