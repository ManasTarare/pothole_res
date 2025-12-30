🛣️ Smart Road Intelligence System
A comprehensive AI-powered platform for automated road inspection, pothole detection, and geospatial risk assessment. This system utilizes Computer Vision (YOLOv8) and GIS technology to detect road defects from video feeds, estimate repair costs, and alert contractors in real-time.

🌟 Features
1. 📊 Smart Dashboard (User)
Video Analysis: Upload inspection footage to automatically detect potholes using a trained YOLOv8 model.

Geospatial Mapping: Select Start/End coordinates via an interactive map to trace the route.

Manual Controls: Override road names and adjust horizon filters to ignore sky/trees in forward-facing cameras.

Risk Visualization: View a color-coded Risk Map and heatmaps of damaged zones.

2. ⚠️ Live Warning System (Driver)
Real-Time Alerts: Simulates a connected vehicle interface that warns drivers of approaching potholes based on GPS location.

Dual Modes:

Known Road: Warns of hazards using the cloud database.

New Road: Maps new terrain in real-time using the live webcam.

3. 🔒 Admin Command Center
Secure Access: Password-protected panel for city planners and engineers.

Global Analytics: View city-wide damage reports, total budget estimates, and critical zones.

Contractor Assignment: Select a specific road and dispatch a work order via SMS (Twilio) to a repair contractor.

Data Management: Selectively delete old records or perform a hard reset of the database.

🛠️ Tech Stack
Core: Python 3.9+, Streamlit

Computer Vision: Ultralytics YOLOv8, OpenCV, Pillow

Geospatial: OSMnx, Folium, Streamlit-Folium, Geopy

Data Handling: Pandas, NumPy, JSON (Simulated Cloud)

Notifications: Twilio API (SMS)

🚀 Installation & Setup
1. Clone the Repository
Bash

git clone https://github.com/your-username/smart-road-intelligence.git
cd smart-road-intelligence
2. Install Dependencies
It is recommended to use a virtual environment.

Bash

pip install -r requirements.txt
Note: If you have issues installing osmnx on Windows, use Conda: conda install -c conda-forge osmnx geopandas

3. Setup the AI Model
Place your trained YOLO model weights file (e.g., best.pt) in the project root directory.

Update DEFAULT_MODEL_PATH in main.py if your filename differs from best.pt.

4. Configuration (Twilio SMS)
To enable real SMS alerts, update the credentials in main.py (approx. line 70):

Python

TWILIO_SID = "AC_YOUR_REAL_SID"
TWILIO_AUTH = "YOUR_REAL_AUTH_TOKEN"
TWILIO_PHONE = "+1_YOUR_TWILIO_NUMBER"
If left as placeholders, the system will simulate SMS in the console.

▶️ How to Run
Run the application using Streamlit:

Bash

streamlit run main.py
🔑 Default Credentials
Admin Username: admin

Admin Password: RoadSafe2025!

📂 Project Structure
Plaintext

├── main.py               # The main application code
├── requirements.txt      # Python dependencies
├── best.pt               # Trained YOLOv8 model weights
├── pothole_db.json       # Local database (Simulated Cloud)
├── README.md             # Documentation
└── runs/                 # (Optional) Training logs if applicable
🔮 Future Improvements
Integration with live GPS hardware modules.

Support for multiple object classes (Cracks, Manholes, Speed Bumps).

Mobile app version for field reporters.

Developed for the Smart City Initiative. 🚦
