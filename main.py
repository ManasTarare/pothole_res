import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import math
import subprocess
import shutil
import os
import torch
import json
import time
from geopy.distance import geodesic 

# GIS & Map Libraries
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, box
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# SMS Library
try:
    from twilio.rest import Client
except ImportError:
    pass 

# =========================
# 1. PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Smart Road Intelligence", 
    layout="wide", 
    page_icon="🛣️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0E1117;
        border-radius: 4px 4px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 2. SYSTEM CONFIG
# =========================
ADMIN_USER = "admin"
ADMIN_PASS = "RoadSafe2025!"
DEFAULT_MODEL_PATH = r"E:\road\runs\detect\pothole_model_final\weights\best.pt" 

# Twilio Credentials (REPLACE WITH YOUR REAL KEYS)
TWILIO_SID = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
TWILIO_AUTH = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_PHONE = "+15555555555"

DIST_THRESHOLD = 50       
FRAME_COOLDOWN = 20       
BBOX_BUFFER_DEG = 0.005
FRAME_SKIP = 3 
CLOUD_DB_FILE = "pothole_db.json"

# =========================
# 3. UTILITIES
# =========================
def has_ffmpeg():
    return shutil.which("ffmpeg") is not None

def get_video_writer(output_path, fps, width, height):
    if has_ffmpeg():
        return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)), "mp4_intermediate"
    else:
        return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (width, height)), "webm"

def convert_video_for_browser(input_path, output_path):
    if has_ffmpeg():
        try:
            command = ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-acodec", "aac", output_path]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return output_path, "video/mp4"
        except: return input_path, "video/mp4"
    return input_path, "video/webm"

def save_to_cloud_db(records, source_id):
    """Saves records to the local JSON file (Simulated Cloud)."""
    new_entries = []
    for r in records:
        new_entries.append({
            "lat": r["Latitude"],
            "lon": r["Longitude"],
            "road_name": r.get("Road_Name", "Unknown Road"), 
            "severity": r["Severity"],
            "cost": r["Est_Cost"],
            "source": source_id
        })
    
    existing_data = []
    if os.path.exists(CLOUD_DB_FILE):
        try:
            with open(CLOUD_DB_FILE, "r") as f:
                existing_data = json.load(f)
        except: pass
    
    existing_data.extend(new_entries)
    
    with open(CLOUD_DB_FILE, "w") as f:
        json.dump(existing_data, f)
    
    return len(new_entries)

def send_sms_alert(to_number, body):
    try:
        if "ACxxx" in TWILIO_SID: 
            st.warning("⚠️ Twilio Credentials not set. SMS Simulated in Console.")
            print(f"--- SMS TO {to_number} ---\n{body}\n-----------------------")
            return True
        client = Client(TWILIO_SID, TWILIO_AUTH)
        client.messages.create(body=body, from_=TWILIO_PHONE, to=to_number)
        return True
    except Exception as e:
        # Hide full error from UI, but print to console for debugging
        print(f"Twilio Error Log: {e}") 
        st.error("❌ Failed to send SMS.")
        return False

# =========================
# 4. SESSION STATE INIT
# =========================
keys = {
    "video_id": None, "video_processed": False, "records": [], 
    "output_video_path": None, "output_mime": None, 
    "road_geom": None, "road_names": [], 
    "snapshots": [], "admin_logged_in": False,
    "scan_trigger": False, 
    "cost_minor": 50, "cost_mod": 150, "cost_sev": 400, "global_conf": 0.25,
    "uploader_key": 0,
    # Map Coordinates Defaults
    "start_lat": 19.0760, "start_lon": 72.8777,
    "end_lat": 19.0800, "end_lon": 72.8800,
    "map_selection_mode": "🟢 Set Start Point" # Default mode
}

for k, v in keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# 5. MODEL LOADER
# =========================
@st.cache_resource
def load_model(path):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(path)
        model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# =========================
# 6. GIS ENGINE
# =========================
@st.cache_data
def fetch_gis_data(start_lat, start_lon, end_lat, end_lon):
    try:
        G = ox.graph_from_point((start_lat, start_lon), dist=3000, network_type="drive")
        start_node = ox.nearest_nodes(G, start_lon, start_lat)
        end_node = ox.nearest_nodes(G, end_lon, end_lat)
        try: route = ox.shortest_path(G, start_node, end_node)
        except: return None, None
        if not route: return None, None

        edges = ox.graph_to_gdfs(G, nodes=False)
        geoms, names = [], []
        
        for u, v in zip(route[:-1], route[1:]):
            try:
                edge = edges.loc[(u, v, 0)] if (u, v, 0) in edges.index else edges.loc[(u, v)]
                if isinstance(edge, pd.DataFrame):
                    geom = edge.iloc[0].geometry
                    name = edge.iloc[0].get('name', 'Unknown Road')
                else:
                    geom = edge.geometry
                    name = edge.get('name', 'Unknown Road')
                
                if isinstance(name, list): name = name[0]
                geoms.append(geom)
                names.append(name)
            except: continue
            
        if not geoms: return None, None
        
        road_geom = LineString([pt for g in geoms for pt in g.coords])
        road_network_data = list(zip(geoms, names))
        return road_geom, road_network_data
    except Exception: return None, None

def get_road_name_at_point(point_geom, network_data):
    if not network_data: return "Unknown Road"
    best_name, min_dist = "Unknown Road", float('inf')
    for geom, name in network_data:
        dist = geom.distance(point_geom)
        if dist < min_dist:
            min_dist = dist
            best_name = str(name)
    return best_name

# =========================
# 7. PAGE FUNCTIONS
# =========================

def sidebar_nav():
    st.sidebar.title("🛣️ Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Live Warnings", "Admin Panel"])
    
    st.sidebar.divider()
    model_path_input = st.sidebar.text_input("Model Path", DEFAULT_MODEL_PATH, disabled=True) 
    model, device_type = load_model(model_path_input)
    
    if device_type == 'cuda':
        st.sidebar.success(f"GPU Active: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("Running on CPU")
        
    return page, model, device_type

def admin_panel():
    st.header("🔒 Admin Command Center")
    if not st.session_state["admin_logged_in"]:
        with st.form("login_form"):
            st.subheader("Authentication Required")
            user = st.text_input("Username")
            passwd = st.text_input("Password", type="password")
            if st.form_submit_button("Unlock System"):
                if user == ADMIN_USER and passwd == ADMIN_PASS:
                    st.session_state["admin_logged_in"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")
        return

    st.success(f"✅ Authenticated as {ADMIN_USER}")
    if st.button("Log Out"):
        st.session_state["admin_logged_in"] = False
        st.rerun()
    
    st.divider()
    t1, t2, t3 = st.tabs(["📊 Global Analytics", "👷 Contractor Assignment", "⚙️ Data & Config"])
    
    # --- TAB 1: ANALYTICS ---
    with t1:
        st.subheader("🌍 City-Wide Road Intelligence")
        if os.path.exists(CLOUD_DB_FILE):
            with open(CLOUD_DB_FILE, "r") as f:
                data = json.load(f)
            
            if not data:
                st.info("Database is empty.")
            else:
                df = pd.DataFrame(data)
                # Normalization
                if 'road_name' not in df.columns: df['road_name'] = 'Unknown Road'
                if 'severity' not in df.columns: df['severity'] = 'Minor'
                if 'cost' not in df.columns: df['cost'] = 0.0
                
                # Top Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Potholes Detected", len(df))
                total_cost = df['cost'].sum()
                m2.metric("Total Repair Budget", f"${total_cost:,.2f}")
                m3.metric("Critical Zones", df['road_name'].nunique())
                st.divider()
                
                # Specific Road Analysis
                st.subheader("🛣️ Road-Specific Breakdown")
                unique_roads = df['road_name'].unique().tolist()
                selected_road_analysis = st.selectbox("Select Road for Detail View:", unique_roads)
                
                if selected_road_analysis:
                    road_df = df[df['road_name'] == selected_road_analysis]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Potholes on Route", len(road_df))
                    c2.metric("Severe Defects", len(road_df[road_df['severity'] == "Severe"]))
                    c3.metric("Est. Repair Cost", f"${road_df['cost'].sum():,.2f}")
                    st.bar_chart(road_df['severity'].value_counts())

    # --- TAB 2: CONTRACTOR ASSIGNMENT ---
    with t2:
        st.subheader("👷 Assign Repair Work Order")
        if os.path.exists(CLOUD_DB_FILE):
            with open(CLOUD_DB_FILE, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            available_roads = df['road_name'].unique().tolist() if 'road_name' in df.columns else []
        else:
            available_roads = []

        with st.form("contractor_form"):
            c1, c2 = st.columns(2)
            cont_name = c1.text_input("Contractor Name")
            cont_phone = c2.text_input("Phone Number (with Country Code)", value="+91")
            cont_corp = st.text_input("Corporation / Agency Name")
            
            target_road = st.selectbox("Select Target Road", available_roads)
            
            submitted = st.form_submit_button("🚀 Dispatch Work Order (SMS)")
            
            if submitted:
                if target_road and cont_phone:
                    road_data = df[df['road_name'] == target_road]
                    count = len(road_data)
                    budget = road_data['cost'].sum() if 'cost' in road_data else 0
                    start_pt = f"{road_data.iloc[0]['lat']:.4f}, {road_data.iloc[0]['lon']:.4f}"
                    
                    sms_body = (
                        f"WORK ORDER: {cont_corp}\n"
                        f"Road: {target_road}\n"
                        f"Defects: {count}\n"
                        f"Budget: ${budget:,.2f}\n"
                        f"Start Loc: {start_pt}\n"
                        f"- Sent via Smart Road AI"
                    )
                    
                    if send_sms_alert(cont_phone, sms_body):
                        st.success(f"✅ Work Order sent to {cont_name} successfully!")
                    
                else:
                    st.warning("Please select a road and enter a phone number.")

    # --- TAB 3: DATA & CONFIG ---
    with t3:
        st.subheader("⚙️ System Configuration")
        st.markdown("##### 💰 Repair Cost Settings")
        c1, c2, c3 = st.columns(3)
        new_minor = c1.number_input("Minor ($)", value=st.session_state["cost_minor"])
        new_mod = c2.number_input("Moderate ($)", value=st.session_state["cost_mod"])
        new_sev = c3.number_input("Severe ($)", value=st.session_state["cost_sev"])
        if st.button("💾 Save Pricing"):
            st.session_state["cost_minor"] = new_minor
            st.session_state["cost_mod"] = new_mod
            st.session_state["cost_sev"] = new_sev
            st.toast("Pricing updated!", icon="✅")

        st.divider()
        st.markdown("##### 🗑️ Database Management")
        if st.button("🔥 HARD RESET (Wipe All Data)", type="primary"):
            if os.path.exists(CLOUD_DB_FILE):
                os.remove(CLOUD_DB_FILE)
                st.error("Database wiped.")
                st.rerun()

def warning_system_tab(model, device):
    st.header("⚠️ Live Road Warning System")
    st.markdown("### 📡 Connected Vehicle Interface")
    
    img_file_buffer = st.camera_input("Scan Road (Live Feed)")
    
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.pi.uint8), cv2.IMREAD_COLOR)
        
        mode = st.radio("System Mode", ["Scenario A: Known Road (Data Exists)", "Scenario B: New Road (Mapping Mode)"], horizontal=True)
        
        if "Scenario A" in mode:
            if not os.path.exists(CLOUD_DB_FILE):
                st.warning("Database empty. Switch to Mapping Mode.")
            else:
                with open(CLOUD_DB_FILE, "r") as f:
                    cloud_data = json.load(f)
                
                if cloud_data:
                    sim_user_lat = cloud_data[0]['lat']
                    sim_user_lon = cloud_data[0]['lon']
                    
                    st.info(f"📍 GPS Locked: {sim_user_lat:.4f}, {sim_user_lon:.4f}")
                    
                    hazards = []
                    for p in cloud_data:
                        dist = geodesic((sim_user_lat, sim_user_lon), (p['lat'], p['lon'])).meters
                        if dist < 500:
                            hazards.append(p)
                    
                    if hazards:
                        st.error(f"🚨 ALERT: {len(hazards)} Potholes detected in next 500m!")
                        m = folium.Map(location=[sim_user_lat, sim_user_lon], zoom_start=17)
                        folium.Marker([sim_user_lat, sim_user_lon], popup="You", icon=folium.Icon(color="blue", icon="car", prefix="fa")).add_to(m)
                        for h in hazards:
                            folium.CircleMarker([h['lat'], h['lon']], radius=6, color="red", fill=True, tooltip=h['severity']).add_to(m)
                        st_folium(m, height=250, use_container_width=True)
                        st.dataframe(pd.DataFrame(hazards)[['road_name', 'severity', 'lat', 'lon']])
                    else:
                        st.success("✅ Road Clear.")

        else:
            st.warning("⚠️ Unknown Territory. Mapping in Real-Time...")
            results = model(cv2_img, conf=0.25, device=device)
            detections = []
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, sc, cls = map(float, box)
                area = (x2-x1)*(y2-y1)
                sev = "Severe" if area > 8000 else ("Moderate" if area > 2000 else "Minor")
                cv2.rectangle(cv2_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                detections.append({
                    "Latitude": 19.0760, 
                    "Longitude": 72.8777,
                    "Road_Name": "Live Scanned Road",
                    "Severity": sev,
                    "Est_Cost": 150
                })
            
            st.image(cv2_img, channels="BGR", caption="Real-Time Analysis")
            if detections:
                save_to_cloud_db(detections, "Live Camera")
                st.toast(f"Saved {len(detections)} new potholes to DB!", icon="💾")

def user_dashboard(model, device_type):
    st.title("🛣️ Smart Road Dashboard")
    conf = st.session_state["global_conf"]
    
    # --- INPUT SECTION ---
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 1. Route Configuration")
        # Manual Overrides
        enable_horizon = st.checkbox("Enable Horizon Filter", value=True)
        horizon_pct = st.slider("Filter Level", 0.0, 1.0, 0.4, 0.05) if enable_horizon else 0.0
        manual_road_name = st.text_input("Road Name", placeholder="e.g. NH-44")
        
        st.divider()
        st.markdown("#### 🗺️ Coordinate Selection")
        
        # --- MAP SELECTION LOGIC ---
        selection_mode = st.radio("Click Map to Set:", ["🟢 Set Start Point", "🔴 Set End Point"], horizontal=True)
        
        # Display Current Coords
        c_lat, c_lon = st.columns(2)
        c_lat.caption(f"Start: {st.session_state.start_lat:.4f}, {st.session_state.start_lon:.4f}")
        c_lon.caption(f"End:   {st.session_state.end_lat:.4f}, {st.session_state.end_lon:.4f}")

        # Interactive Map
        m = folium.Map(location=[st.session_state.start_lat, st.session_state.start_lon], zoom_start=13)
        
        # Draw Markers
        folium.Marker(
            [st.session_state.start_lat, st.session_state.start_lon], 
            popup="Start", 
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        
        folium.Marker(
            [st.session_state.end_lat, st.session_state.end_lon], 
            popup="End", 
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)
        
        # Draw connecting line
        folium.PolyLine(
            [(st.session_state.start_lat, st.session_state.start_lon), 
             (st.session_state.end_lat, st.session_state.end_lon)],
            color="blue", weight=2, opacity=0.5, dash_array='5, 5'
        ).add_to(m)

        m.add_child(folium.LatLngPopup()) 
        
        map_out = st_folium(m, height=300, width=400)
        
        # Handle Clicks
        if map_out.get("last_clicked"):
            coords = map_out["last_clicked"]
            if "Start Point" in selection_mode:
                if coords['lat'] != st.session_state.start_lat:
                    st.session_state.start_lat = coords['lat']
                    st.session_state.start_lon = coords['lng']
                    st.rerun()
            elif "End Point" in selection_mode:
                if coords['lat'] != st.session_state.end_lat:
                    st.session_state.end_lat = coords['lat']
                    st.session_state.end_lon = coords['lng']
                    st.rerun()

    with c2:
        st.markdown("### 2. Upload & Analyze")
        vid_file = st.file_uploader("Upload Video", ["mp4", "avi"], key=f"uploader_{st.session_state.uploader_key}")
        
        if vid_file:
            if st.button("🚀 Start Analysis", type="primary"):
                st.session_state.video_id = vid_file.name
                
                with st.spinner("🗺️ Fetching GIS Data..."):
                    road_geom, road_names_data = fetch_gis_data(st.session_state.start_lat, st.session_state.start_lon, st.session_state.end_lat, st.session_state.end_lon)
                    st.session_state.road_geom = road_geom
                    st.session_state.road_names = road_names_data

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(vid_file.read())
                    input_path = tmp.name

                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w, h = int(cap.get(3)), int(cap.get(4))
                horizon_y = int(h * horizon_pct)
                
                if has_ffmpeg():
                    raw_out = os.path.join(tempfile.gettempdir(), "raw_temp.mp4")
                    out, method = get_video_writer(raw_out, fps, w, h)
                else:
                    raw_out = os.path.join(tempfile.gettempdir(), "raw_temp.webm")
                    out, method = get_video_writer(raw_out, fps, w, h)

                tracked = []
                frame_idx = 0
                prog_bar = st.progress(0)
                status_txt = st.empty()
                st.session_state.records = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    if frame_idx % FRAME_SKIP != 0:
                        out.write(frame)
                        frame_idx += 1
                        continue

                    results = model(frame, conf=conf, device=device_type, verbose=False)
                    
                    if enable_horizon:
                        cv2.line(frame, (0, horizon_y), (w, horizon_y), (255, 100, 0), 3)
                    
                    for box_dat in results[0].boxes.data.cpu().numpy(): 
                        x1, y1, x2, y2, sc, cls = map(float, box_dat)
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        area = (x2-x1)*(y2-y1)

                        if enable_horizon and cy < horizon_y: continue 

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        is_new = True
                        for t in tracked:
                            dist = math.sqrt((cx-t[0])**2 + (cy-t[1])**2)
                            if dist < DIST_THRESHOLD and (frame_idx - t[2]) < FRAME_COOLDOWN:
                                is_new = False; break
                        
                        if is_new:
                            tracked.append((cx, cy, frame_idx))
                            
                            final_name = "Unknown Road"
                            if manual_road_name:
                                final_name = manual_road_name
                                pct = frame_idx / total_frames if total_frames > 0 else 0
                                lat = st.session_state.start_lat + (st.session_state.end_lat - st.session_state.start_lat) * pct
                                lon = st.session_state.start_lon + (st.session_state.end_lon - st.session_state.start_lon) * pct
                            else:
                                if st.session_state.road_geom:
                                    pct = frame_idx / total_frames if total_frames > 0 else 0
                                    pt = st.session_state.road_geom.interpolate(pct, normalized=True)
                                    lat, lon = pt.y, pt.x
                                    if st.session_state.road_names:
                                        final_name = get_road_name_at_point(pt, st.session_state.road_names)
                                else:
                                    lat, lon = st.session_state.start_lat, st.session_state.start_lon

                            sev = "Severe" if area > 8000 else ("Moderate" if area > 2000 else "Minor")
                            
                            cost_map = {
                                "Minor": st.session_state["cost_minor"],
                                "Moderate": st.session_state["cost_mod"],
                                "Severe": st.session_state["cost_sev"]
                            }

                            st.session_state.records.append({
                                "Latitude": lat, "Longitude": lon,
                                "Road_Name": final_name,
                                "Severity": sev, "Est_Cost": cost_map[sev]
                            })

                    out.write(frame)
                    frame_idx += 1
                    if frame_idx % 20 == 0:
                        prog_bar.progress(min(frame_idx/total_frames, 1.0))
                        status_txt.text(f"Processing... {int((frame_idx/total_frames)*100)}%")

                cap.release()
                out.release()
                
                save_to_cloud_db(st.session_state.records, st.session_state.video_id)
                final_path = os.path.join(tempfile.gettempdir(), "final_output_web.mp4")
                final_path, mime = convert_video_for_browser(raw_out, final_path)
                st.session_state.output_video_path = final_path
                st.session_state.output_mime = mime
                st.session_state.video_processed = True
                st.rerun()

    # --- RESULTS SECTION ---
    if st.session_state.video_processed:
        st.divider()
        st.subheader("🏁 Analysis Results")
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown("**Processed Video**")
            try:
                with open(st.session_state.output_video_path, "rb") as f:
                    st.video(f.read(), format=st.session_state.output_mime)
            except: st.error("Video Error")
        
        with c_res2:
            st.markdown("**Risk Map**")
            df = pd.DataFrame(st.session_state.records)
            if not df.empty and st.session_state.road_geom:
                 m = folium.Map(location=[st.session_state.start_lat, st.session_state.start_lon], zoom_start=14)
                 folium.GeoJson(st.session_state.road_geom, style_function=lambda x: {'color':'blue','weight':4}).add_to(m)
                 HeatMap(df[['Latitude', 'Longitude']].values, radius=15).add_to(m)
                 st_folium(m, height=350, use_container_width=True)
            else:
                st.info("No geospatial data available.")

        if st.button("🔄 Reset System", type="primary"):
            st.session_state["video_id"] = None
            st.session_state["video_processed"] = False
            st.session_state["records"] = []
            st.session_state["road_geom"] = None
            st.session_state["uploader_key"] += 1
            st.rerun()

# =========================
# 8. APP EXECUTION
# =========================
selected_page, model, device = sidebar_nav()

if model is None:
    st.error("⚠️ Model not found. Please check path in Admin Panel.")
else:
    if selected_page == "Dashboard":
        user_dashboard(model, device)
    elif selected_page == "Live Warnings":
        warning_system_tab(model, device)
    elif selected_page == "Admin Panel":
        admin_panel()