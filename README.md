🛣️ Smart Road Intelligence System
A comprehensive AI-powered platform for automated road inspection, pothole detection, and geospatial risk assessment. Utilises Computer Vision (YOLOv8) and GIS technology to detect road defects from video feeds, estimate repair costs, and alert contractors in real-time.

🌟 Features
1. 📊 Smart Dashboard (User)

Video Analysis — Upload inspection footage to automatically detect potholes using a trained YOLOv8 model.
Geospatial Mapping — Select Start/End coordinates via an interactive map to trace the route.
Manual Controls — Override road names and adjust the horizon filter to ignore sky/trees in forward-facing cameras.
Risk Visualisation — View a colour-coded Risk Map and heatmap of damaged zones.

2. ⚠️ Live Warning System (Driver)

Real-Time Alerts — Simulates a connected-vehicle interface warning drivers of approaching potholes based on GPS location.
Dual Modes:

Known Road — Warns of hazards using the cloud database.
New Road — Maps new terrain in real-time using the live webcam (local deployment only).



3. 🔒 Admin Command Center

Secure Access — Password-protected panel for city planners and engineers.
Global Analytics — City-wide damage reports, total budget estimates, and critical zones.
Contractor Assignment — Select a road and dispatch a work order via SMS (Twilio).
Data Management — Reset database or selectively review records.


🛠️ Tech Stack
LayerTechnologyFrontendPython 3.9+, StreamlitComputer VisionUltralytics YOLOv8, OpenCV, PillowGeospatialOSMnx, Folium, Streamlit-Folium, GeopyData StorageSQLite (via database.py)NotificationsTwilio API (SMS)Configpython-dotenv

🚀 Local Installation & Setup
1. Clone the Repository
bashgit clone https://github.com/your-username/smart-road-intelligence.git
cd smart-road-intelligence
2. Create a Virtual Environment
bashpython -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

Note for Windows/osmnx issues: use Conda instead:
bashconda install -c conda-forge osmnx geopandas

3. Configure Environment Variables
bashcp .env.example .env
# Edit .env and fill in your values
Key variables:
VariableDescriptionMODEL_PATHPath to your YOLO weights file (e.g. best.pt)ADMIN_USERAdmin login usernameADMIN_PASSAdmin login passwordTWILIO_SID / TWILIO_AUTH / TWILIO_PHONETwilio credentials (leave blank to simulate)DATABASE_PATHSQLite database file path (default: pothole.db)
4. Add Your Model Weights
Place your trained YOLO weights (e.g. best.pt or yolov8n.pt) in the project root and set MODEL_PATH in .env.
5. Run Locally
bashstreamlit run main.py

☁️ Deploy to Render
This project ships with a render.yaml for one-click deployment.
Steps

Push to GitHub — ensure render.yaml is in the root.
Create a new Web Service on render.com and connect your repo.
Render auto-detects render.yaml and configures the build/start commands.
Set secret environment variables in the Render dashboard → Environment tab:

ADMIN_USER, ADMIN_PASS
TWILIO_SID, TWILIO_AUTH, TWILIO_PHONE (optional)
MODEL_PATH — if you store your weights in the repo, set this to yolov8n.pt or your filename.


Deploy — Render will install dependencies and start Streamlit automatically.


Important: Render's free tier filesystem is ephemeral. The database is written to /tmp/pothole.db (set via DATABASE_PATH). For persistent storage, upgrade to a paid plan with a persistent disk or connect a PostgreSQL database.


Webcam / Live Warning: Cloud platforms do not expose webcam hardware. The Live Warning tab automatically falls back to a database-based hazard viewer when deployed to Render.

Render Environment Variables to Set
IS_CLOUD=true
MODEL_PATH=yolov8n.pt
DATABASE_PATH=/tmp/pothole.db
ADMIN_USER=<your-username>
ADMIN_PASS=<your-secure-password>
SECRET_KEY=<random-long-string>

📂 Project Structure
├── main.py           # Streamlit application (all pages)
├── config.py         # Environment-based configuration
├── database.py       # SQLite data layer
├── requirements.txt  # Python dependencies
├── render.yaml       # Render deployment manifest
├── Procfile          # Start command for PaaS platforms
├── .env.example      # Environment variable template
├── yolov8n.pt        # Fallback YOLO nano weights
├── pothole_db.json   # Legacy JSON data (auto-migrated on first run)
└── README.md

🔑 Default Credentials

Set via environment variables. The fallback values below are for local development only — always change these before deploying.


Admin Username: admin
Admin Password: (set ADMIN_PASS in .env)


🔮 Future Improvements

Integration with live GPS hardware modules.
Support for multiple defect classes (cracks, manholes, speed bumps).
Mobile app for field reporters.
PostgreSQL backend for persistent cloud storage.
OAuth / SSO for admin authentication.
