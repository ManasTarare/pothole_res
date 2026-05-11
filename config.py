"""
Application configuration — all settings loaded from environment variables.
Set these in a .env file locally, or in Render's Environment tab for production.
"""
import os
import secrets
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Security ---
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    # --- Model ---
    MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")

    # --- Database ---
    DATABASE_PATH    = os.environ.get("DATABASE_PATH",    "pothole.db")
    LEGACY_JSON_PATH = os.environ.get("LEGACY_JSON_PATH", "pothole_db.json")

    # --- Admin ---
    # ⚠️  NEVER commit real passwords. Set ADMIN_PASS in .env or Render dashboard.
    ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
    ADMIN_PASS = os.environ.get("ADMIN_PASS", "")   # Empty → admin endpoints locked until set

    # --- Twilio SMS ---
    TWILIO_SID   = os.environ.get("TWILIO_SID",   "")
    TWILIO_AUTH  = os.environ.get("TWILIO_AUTH",  "")
    TWILIO_PHONE = os.environ.get("TWILIO_PHONE", "")

    # --- Detection Tuning ---
    DIST_THRESHOLD     = int(os.environ.get("DIST_THRESHOLD",   "50"))
    FRAME_COOLDOWN     = int(os.environ.get("FRAME_COOLDOWN",   "20"))
    FRAME_SKIP         = int(os.environ.get("FRAME_SKIP",       "3"))
    DEFAULT_CONFIDENCE = float(os.environ.get("DEFAULT_CONFIDENCE", "0.25"))

    # --- Severity Cost Defaults (in ₹) ---
    COST_MINOR    = float(os.environ.get("COST_MINOR",    "50"))
    COST_MODERATE = float(os.environ.get("COST_MODERATE", "150"))
    COST_SEVERE   = float(os.environ.get("COST_SEVERE",   "400"))

    # --- Upload ---
    UPLOAD_FOLDER      = os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB

    # --- Server ---
    PORT  = int(os.environ.get("PORT", "8501"))
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    def validate(self):
        """Warn about dangerous default configurations at startup."""
        import logging
        log = logging.getLogger(__name__)
        if not self.ADMIN_PASS:
            log.warning(
                "ADMIN_PASS is not set! Admin endpoints will return 403. "
                "Set ADMIN_PASS in your .env file or Render dashboard."
            )
