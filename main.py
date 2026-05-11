"""
🛣️ Smart Road Intelligence System
REST API entry point — Flask + Gunicorn, suitable for Render deployment.

Endpoints:
  GET  /health                     — liveness check
  POST /api/analyze                — upload video + route coords, returns detections
  GET  /api/records                — all pothole records (paginated)
  GET  /api/stats                  — aggregate statistics
  GET  /api/roads                  — unique road names
  GET  /api/roads/<name>           — records for a specific road
  POST /api/dispatch               — send SMS work-order to contractor (requires X-Admin-Key)
  POST /api/admin/reset            — wipe all records (requires X-Admin-Key)
  POST /api/admin/pricing          — update repair cost thresholds (requires X-Admin-Key)
  GET  /api/video/<filename>       — stream processed video file
  GET  /api/hazards                — potholes near a GPS coordinate
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import logging

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory, abort
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import pandas as pd

# GIS
import osmnx as ox
import geopandas as gpd

# SMS (optional)
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

from config import Config
from database import Database

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=".", static_url_path="")
cfg = Config()
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH  # 500 MB upload limit
db = Database(cfg.DATABASE_PATH)

# One-time JSON migration
if os.path.exists(cfg.LEGACY_JSON_PATH):
    migrated = db.import_from_json(cfg.LEGACY_JSON_PATH)
    if migrated:
        try:
            os.rename(cfg.LEGACY_JSON_PATH, cfg.LEGACY_JSON_PATH + ".imported")
            log.info(f"Migrated {migrated} records from legacy JSON.")
        except OSError as e:
            log.warning(f"Could not rename legacy JSON: {e}")

# In-memory pricing (overridable via /api/admin/pricing)
_pricing = {
    "Minor":    cfg.COST_MINOR,
    "Moderate": cfg.COST_MODERATE,
    "Severe":   cfg.COST_SEVERE,
}

# Allowed video MIME types / extensions
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_filename(name: str) -> str:
    """Strip path separators and unsafe chars from a filename, then add a UUID prefix."""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))
    return f"{uuid.uuid4().hex}_{base}"


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def get_video_writer(output_path: str, fps: float, width: int, height: int):
    if has_ffmpeg():
        return (
            cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)),
            "mp4_intermediate",
        )
    webm_path = output_path.replace(".mp4", ".webm")
    return (
        cv2.VideoWriter(webm_path, cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height)),
        "webm",
    )


def convert_video_for_browser(input_path: str, output_path: str):
    if has_ffmpeg():
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-acodec", "aac", output_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            return output_path, "video/mp4"
        except subprocess.CalledProcessError as e:
            log.warning(f"FFmpeg conversion failed: {e.stderr.decode(errors='ignore')}")
    if input_path.endswith(".webm"):
        return input_path, "video/webm"
    return input_path, "video/mp4"


def send_sms_alert(to_number: str, body: str) -> bool:
    if not TWILIO_AVAILABLE:
        log.info(f"[SMS SIM — Twilio not installed] TO {to_number}:\n{body}")
        return True
    if not cfg.TWILIO_SID or cfg.TWILIO_SID.startswith("AC_YOUR"):
        log.info(f"[SMS SIM — credentials not set] TO {to_number}:\n{body}")
        return True
    try:
        client = TwilioClient(cfg.TWILIO_SID, cfg.TWILIO_AUTH)
        client.messages.create(body=body, from_=cfg.TWILIO_PHONE, to=to_number)
        return True
    except Exception as e:
        log.error(f"[Twilio Error] {e}")
        return False


# ---------------------------------------------------------------------------
# Model loader (cached at process level)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def load_model(path: str):
    """Load and cache a YOLO model. Returns (model, device) or (None, None) on failure."""
    if path not in _model_cache:
        try:
            from ultralytics import YOLO
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(path)
            model.to(device)
            _model_cache[path] = (model, device)
            log.info(f"Model loaded from '{path}' on {device}.")
        except Exception as e:
            log.error(f"[Model Load Error] {e}")
            _model_cache[path] = (None, None)
    return _model_cache[path]


# ---------------------------------------------------------------------------
# GIS engine
# ---------------------------------------------------------------------------

def fetch_gis_data(start_lat, start_lon, end_lat, end_lon):
    try:
        G = ox.graph_from_point((start_lat, start_lon), dist=3000, network_type="drive")
        start_node = ox.nearest_nodes(G, start_lon, start_lat)
        end_node   = ox.nearest_nodes(G, end_lon,   end_lat)
        try:
            route = ox.shortest_path(G, start_node, end_node)
        except Exception:
            return None, None
        if not route:
            return None, None

        edges = ox.graph_to_gdfs(G, nodes=False)
        geoms, names = [], []

        for u, v in zip(route[:-1], route[1:]):
            try:
                edge = (
                    edges.loc[(u, v, 0)]
                    if (u, v, 0) in edges.index
                    else edges.loc[(u, v)]
                )
                if isinstance(edge, pd.DataFrame):
                    geom = edge.iloc[0].geometry
                    name = edge.iloc[0].get("name", "Unknown Road")
                else:
                    geom = edge.geometry
                    name = edge.get("name", "Unknown Road")
                if isinstance(name, list):
                    name = name[0]
                geoms.append(geom)
                names.append(str(name) if name else "Unknown Road")
            except Exception:
                continue

        if not geoms:
            return None, None

        road_geom = LineString([pt for g in geoms for pt in g.coords])
        return road_geom, list(zip(geoms, names))
    except Exception as e:
        log.warning(f"GIS fetch failed: {e}")
        return None, None


def get_road_name_at_point(point_geom: Point, network_data) -> str:
    if not network_data:
        return "Unknown Road"
    best_name, min_dist = "Unknown Road", float("inf")
    for geom, name in network_data:
        d = geom.distance(point_geom)
        if d < min_dist:
            min_dist, best_name = d, str(name)
    return best_name


# ---------------------------------------------------------------------------
# Admin auth helper
# ---------------------------------------------------------------------------

def _require_admin():
    """Abort 403 if the X-Admin-Key header doesn't match the configured password."""
    key = request.headers.get("X-Admin-Key", "")
    if not key or key != cfg.ADMIN_PASS:
        abort(403, description="Invalid or missing X-Admin-Key header.")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e.description)}), 400

@app.errorhandler(403)
def forbidden(e):
    return jsonify({"error": str(e.description)}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": str(e.description)}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 500 MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error. Check server logs."}), 500


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def root():
    """Serve the HTML frontend from the project root (no templates/ folder needed)."""
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    model, _ = load_model(cfg.MODEL_PATH)
    return jsonify({
        "status": "ok",
        "model_path": cfg.MODEL_PATH,
        "model_loaded": model is not None,
        "db_path": cfg.DATABASE_PATH,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Multipart form fields:
      video        — video file (mp4 / avi / mov / mkv / webm)
      start_lat    — float
      start_lon    — float
      end_lat      — float
      end_lon      — float
      confidence   — float  (optional, default from config)
      road_name    — string (optional override)
      horizon      — bool   (optional, default true)
      horizon_pct  — float  (optional, default 0.4)

    Returns JSON with detections list and output_video URL.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded. Use field name 'video'."}), 400

    vid_file = request.files["video"]
    if not vid_file.filename:
        return jsonify({"error": "Uploaded file has no filename."}), 400

    # Validate extension
    ext = os.path.splitext(vid_file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}"}), 400

    # Parse form params with defaults
    try:
        start_lat   = float(request.form.get("start_lat",  19.0760))
        start_lon   = float(request.form.get("start_lon",  72.8777))
        end_lat     = float(request.form.get("end_lat",    19.0800))
        end_lon     = float(request.form.get("end_lon",    72.8800))
        conf        = float(request.form.get("confidence", cfg.DEFAULT_CONFIDENCE))
        horizon_pct = float(request.form.get("horizon_pct", 0.4))
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid numeric parameter: {e}"}), 400

    manual_road      = request.form.get("road_name", "").strip()
    enable_horizon   = request.form.get("horizon", "true").lower() != "false"

    # Clamp confidence
    conf = max(0.01, min(conf, 1.0))

    model, device_type = load_model(cfg.MODEL_PATH)
    if model is None:
        return jsonify({"error": f"Model could not be loaded from '{cfg.MODEL_PATH}'. Check MODEL_PATH."}), 500

    safe_name = _safe_filename(vid_file.filename)
    tmp_dir   = tempfile.gettempdir()
    input_path = os.path.join(tmp_dir, f"input_{safe_name}")
    raw_out    = os.path.join(tmp_dir, f"raw_{safe_name}.mp4")
    final_out  = os.path.join(tmp_dir, f"final_{safe_name}.mp4")

    try:
        vid_file.save(input_path)

        # Fetch GIS data (non-fatal if it fails)
        road_geom, road_names_data = fetch_gis_data(start_lat, start_lon, end_lat, end_lon)

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open uploaded video. Check codec/format."}), 422

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        horizon_y    = int(h * horizon_pct)

        if w == 0 or h == 0:
            cap.release()
            return jsonify({"error": "Video has zero dimensions — possibly corrupted."}), 422

        writer, writer_type = get_video_writer(raw_out, fps, w, h)

        tracked   = []
        records   = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % cfg.FRAME_SKIP != 0:
                writer.write(frame)
                frame_idx += 1
                continue

            # Run inference — do NOT pass device kwarg here; it was set at load time
            results = model(frame, conf=conf, verbose=False)

            if enable_horizon:
                cv2.line(frame, (0, horizon_y), (w, horizon_y), (255, 100, 0), 3)

            for box_dat in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, sc, cls = map(float, box_dat)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                area   = (x2 - x1) * (y2 - y1)

                if enable_horizon and cy < horizon_y:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{sc:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                is_new = not any(
                    math.sqrt((cx - t[0]) ** 2 + (cy - t[1]) ** 2) < cfg.DIST_THRESHOLD
                    and (frame_idx - t[2]) < cfg.FRAME_COOLDOWN
                    for t in tracked
                )

                if is_new:
                    tracked.append((cx, cy, frame_idx))
                    pct = frame_idx / total_frames

                    if manual_road:
                        final_name = manual_road
                        lat = start_lat + (end_lat - start_lat) * pct
                        lon = start_lon + (end_lon - start_lon) * pct
                    elif road_geom:
                        pt  = road_geom.interpolate(pct, normalized=True)
                        lat, lon = pt.y, pt.x
                        final_name = get_road_name_at_point(pt, road_names_data)
                    else:
                        lat, lon   = start_lat, start_lon
                        final_name = "Unknown Road"

                    sev = "Severe" if area > 8000 else ("Moderate" if area > 2000 else "Minor")
                    records.append({
                        "lat":       lat,
                        "lon":       lon,
                        "road_name": final_name,
                        "severity":  sev,
                        "cost":      _pricing[sev],
                    })

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

    except Exception as e:
        log.exception("Error during video analysis")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        # Always clean up input temp file
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

    # Persist to DB
    if records:
        db.add_potholes_batch(records, vid_file.filename)

    # Convert for browser playback
    final_path, mime = convert_video_for_browser(raw_out, final_out)
    video_url = f"/api/video/{os.path.basename(final_path)}"

    return jsonify({
        "video_id":         safe_name,
        "original_filename": vid_file.filename,
        "total_detections": len(records),
        "records":          records,
        "output_video_url": video_url,
        "output_mime":      mime,
    })


@app.route("/api/video/<filename>", methods=["GET"])
def serve_video(filename: str):
    # Prevent path traversal
    filename = os.path.basename(filename)
    path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.isfile(path):
        abort(404, description="Video not found or has expired.")
    mime = "video/webm" if filename.endswith(".webm") else "video/mp4"
    return send_file(path, mimetype=mime, conditional=True)


@app.route("/api/records", methods=["GET"])
def get_records():
    try:
        page  = max(1, int(request.args.get("page",  1)))
        limit = max(1, min(int(request.args.get("limit", 100)), 1000))
    except (TypeError, ValueError):
        return jsonify({"error": "page and limit must be integers."}), 400

    rows  = db.get_all()
    start = (page - 1) * limit
    return jsonify({
        "page":    page,
        "limit":   limit,
        "total":   len(rows),
        "records": rows[start: start + limit],
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(db.get_stats())


@app.route("/api/roads", methods=["GET"])
def get_roads():
    return jsonify({"roads": db.get_unique_roads()})


@app.route("/api/roads/<path:road_name>", methods=["GET"])
def get_road_detail(road_name: str):
    rows = db.get_by_road(road_name)
    if not rows:
        return jsonify({"error": f"No records found for road: '{road_name}'."}), 404
    df = pd.DataFrame(rows)
    return jsonify({
        "road_name":    road_name,
        "total":        len(rows),
        "severe_count": int((df["severity"] == "Severe").sum()),
        "total_cost":   float(df["cost"].sum()),
        "records":      rows,
    })


@app.route("/api/dispatch", methods=["POST"])
def dispatch():
    """
    JSON body:
      contractor_name  — str
      phone            — str  (E.164 format, e.g. +919876543210)
      corporation      — str
      road_name        — str
    Requires X-Admin-Key header.
    """
    _require_admin()
    data = request.get_json(force=True) or {}

    road_name   = data.get("road_name",        "").strip()
    phone       = data.get("phone",            "").strip()
    cont_name   = data.get("contractor_name", "Contractor")
    corporation = data.get("corporation",     "")

    if not road_name or not phone:
        return jsonify({"error": "Both 'road_name' and 'phone' fields are required."}), 400

    # Basic E.164 format check
    if not re.match(r"^\+\d{7,15}$", phone):
        return jsonify({"error": "Phone must be in E.164 format, e.g. +919876543210."}), 400

    road_rows = db.get_by_road(road_name)
    if not road_rows:
        return jsonify({"error": f"No records found for road: '{road_name}'."}), 404

    road_df  = pd.DataFrame(road_rows)
    count    = len(road_df)
    budget   = road_df["cost"].sum()
    start_pt = (
        f"{road_df.iloc[0]['lat']:.4f}, {road_df.iloc[0]['lon']:.4f}"
        if count else "N/A"
    )

    sms_body = (
        f"WORK ORDER: {corporation}\n"
        f"Contractor: {cont_name}\n"
        f"Road: {road_name}\n"
        f"Defects: {count}\n"
        f"Budget: \u20b9{budget:,.2f}\n"
        f"Start: {start_pt}\n"
        f"- Smart Road AI"
    )

    ok = send_sms_alert(phone, sms_body)
    if ok:
        return jsonify({"status": "dispatched", "sms_body": sms_body})
    return jsonify({"error": "SMS delivery failed. Check server logs for details."}), 500


@app.route("/api/admin/reset", methods=["POST"])
def admin_reset():
    _require_admin()
    db.reset()
    log.warning("Database reset triggered via API.")
    return jsonify({"status": "reset", "message": "All pothole records deleted."})


@app.route("/api/admin/pricing", methods=["POST"])
def admin_pricing():
    """
    JSON body:
      minor    — float
      moderate — float
      severe   — float
    Requires X-Admin-Key header.
    """
    _require_admin()
    data = request.get_json(force=True) or {}
    errors = []
    for key, field in [("minor", "Minor"), ("moderate", "Moderate"), ("severe", "Severe")]:
        if key in data:
            try:
                val = float(data[key])
                if val < 0:
                    errors.append(f"'{key}' must be non-negative.")
                else:
                    _pricing[field] = val
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number.")
    if errors:
        return jsonify({"error": errors}), 400
    return jsonify({"status": "updated", "pricing": _pricing})


@app.route("/api/hazards", methods=["GET"])
def get_hazards():
    """
    Query params: lat, lon, radius_m (default 500)
    Returns potholes within radius_m metres of the given point.
    """
    try:
        lat      = float(request.args["lat"])
        lon      = float(request.args["lon"])
        radius_m = float(request.args.get("radius_m", 500))
    except (KeyError, ValueError):
        return jsonify({"error": "'lat' and 'lon' query params are required and must be floats."}), 400

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return jsonify({"error": "lat must be in [-90,90] and lon in [-180,180]."}), 400

    all_rows = db.get_all()
    hazards  = [
        p for p in all_rows
        if geodesic((lat, lon), (p["lat"], p["lon"])).meters < radius_m
    ]
    return jsonify({
        "lat":      lat,
        "lon":      lon,
        "radius_m": radius_m,
        "count":    len(hazards),
        "hazards":  hazards,
    })


# ---------------------------------------------------------------------------
# Entry point (local dev only — Render uses Gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting Smart Road Intelligence on port {port}")
    app.run(host="0.0.0.0", port=port, debug=cfg.DEBUG)
