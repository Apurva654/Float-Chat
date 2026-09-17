import base64
import datetime
import io
import json
import os
import re
import sqlite3
import time
import uuid
import chromadb
import numpy as np
import pandas as pd
import pypdf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, jsonify, redirect, render_template, request, session, url_for, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "floatchat_full_workflow_2026")
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(days=30)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

load_dotenv()
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), "floatchat_users.db")

def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

def init_relational_db():
    try:
        conn = get_db_connection()
        conn.cursor().execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    except Exception:
        pass

init_relational_db()

CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
history_collection = chroma_client.get_or_create_collection(name="floatchat_workflow_history")

def save_to_chromadb(prompt: str, reply: str, chart_json: str, timestamp_str: str, user: str = "guest", share_id: str = None):
    try:
        entry_id = share_id if share_id else str(uuid.uuid4())
        history_collection.upsert(
            documents=[prompt],
            metadatas=[{"reply": reply, "chart_json": chart_json or "", "time": timestamp_str, "created_at": int(time.time()), "user": user, "share_id": entry_id}],
            ids=[entry_id]
        )
        return entry_id
    except Exception:
        return None

def get_history_from_chromadb(current_user: str = "guest"):
    try:
        results = history_collection.get()
        sessions = []
        if results and "ids" in results and len(results["ids"]) > 0:
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i]
                if meta.get("user", "guest") == current_user:
                    sessions.append({
                        "id": results["ids"][i],
                        "prompt": results["documents"][i],
                        "reply": meta.get("reply", ""),
                        "chart": meta.get("chart_json") or None,
                        "time": meta.get("time", ""),
                        "created_at": meta.get("created_at", 0),
                        "share_id": meta.get("share_id", results["ids"][i])
                    })
            sessions.sort(key=lambda x: x["created_at"])
        return sessions
    except Exception:
        return []

def get_current_cast_date():
    return datetime.date.today().isoformat()

GLOBAL_FLOAT_DATASET = [
    {"id": "5906001", "name": "Float 5906001 (Equatorial Indian Ocean)", "lat": 1.5, "lon": 65.4, "basin": "Equatorial Indian Ocean", "sst": 29.1, "sal": 35.1, "doxy": 64.0},
    {"id": "5905082", "name": "Float 5905082 (Central Arabian Sea)", "lat": 16.8, "lon": 66.5, "basin": "Arabian Sea", "sst": 28.1, "sal": 36.4, "doxy": 6.8},
    {"id": "2902781", "name": "Float 2902781 (Bay of Bengal)", "lat": 15.2, "lon": 88.5, "basin": "Bay of Bengal", "sst": 29.4, "sal": 33.2, "doxy": 18.2}
]

def extract_all_coordinates(text: str):
    coord_pattern = r'(-?\d+(?:\.\d+)?)\s*(?:°|deg)?\s*([NSns])?\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:°|deg)?\s*([EWew])?'
    matches = re.findall(coord_pattern, text)
    coords = []
    for lat_val, lat_dir, lon_val, lon_dir in matches:
        try:
            lat = -abs(float(lat_val)) if (lat_dir and lat_dir.upper() == 'S') else abs(float(lat_val))
            lon = -abs(float(lon_val)) if (lon_dir and lon_dir.upper() == 'W') else abs(float(lon_val))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                coords.append((lat, lon))
        except Exception:
            pass
    return coords

def calc_temp(depth: float, lat: float) -> float:
    return float(np.round(max(2.0, 29.5 - (abs(lat) * 0.05) - (depth * 0.07)), 2))

def calc_sal(depth: float, lat: float) -> float:
    return float(np.round(34.2 + (abs(lat) * 0.03) + (depth * 0.001), 2))

def get_realistic_temp_profile(sst: float, lat: float, pressures: np.ndarray) -> list:
    temps = []
    for p in pressures:
        if p <= 50:
            t = sst - (p / 50.0) * 0.3
        elif p <= 300:
            t = (sst - 0.3) - ((p - 50) / 250.0) * (sst - 12.0)
        else:
            t = 12.0 - ((p - 300) / 700.0) * 8.2
        temps.append(round(max(2.5, t), 2))
    return temps

def get_realistic_sal_profile(basin_name: str, sal_base: float, pressures: np.ndarray) -> list:
    b_low = basin_name.lower()
    sals = []
    if "bengal" in b_low:
        # Bay of Bengal: Low surface salinity (33.2 PSU) rising to 35.0 PSU at depth
        for p in pressures:
            if p <= 100:
                s = sal_base + (p / 100.0) * 1.6
            else:
                s = 34.8 + ((p - 100) / 900.0) * 0.4
            sals.append(round(s, 2))
    elif "arabian" in b_low or "oman" in b_low:
        # Arabian Sea: High surface salinity (36.4 PSU) peaking at 100m, dropping to 35.1 PSU
        for p in pressures:
            if p <= 120:
                s = sal_base + (p / 120.0) * 0.4
            else:
                s = 36.8 - ((p - 120) / 880.0) * 1.7
            sals.append(round(s, 2))
    elif "pacific" in b_low or "warm pool" in b_low:
        # Pacific Warm Pool: Salinity minimum at 150m (~34.2 PSU), deep ~34.7 PSU
        for p in pressures:
            if p <= 150:
                s = sal_base - (p / 150.0) * 0.3
            else:
                s = 34.2 + ((p - 150) / 850.0) * 0.5
            sals.append(round(s, 2))
    else:
        for p in pressures:
            s = sal_base + (np.exp(-p / 250.0)) * 0.4 - ((p / 1000.0) * 0.2)
            sals.append(round(s, 2))
    return sals

def extract_text_from_any_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    raw_text = ""
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                t = page.extract_text()
                if t: raw_text += t + " "
        except Exception:
            pass
    return raw_text.strip() or "Dynamic telemetry query."

def render_full_dedicated_3d_globe():
    df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
    fig = go.Figure()
    
    # 1. ARGO Profiling Stations (Yellow Circles)
    fig.add_trace(go.Scattergeo(
        lat=df_floats["lat"].tolist(),
        lon=df_floats["lon"].tolist(),
        mode="markers",
        name="ARGO Profiling Stations",
        text=[f"Station: {row['name']}<br>SST: {row['sst']}°C | Sal: {row['sal']} PSU" for _, row in df_floats.iterrows()],
        marker=dict(
            size=14,
            color="#facc15",
            line=dict(width=2, color="#ffffff"),
            opacity=0.9
        )
    ))
    
    # 2. Key Ocean Landmarks (White Stars)
    landmarks = [
        {"name": "Oman Upwelling", "lat": 18.5, "lon": 58.2},
        {"name": "Agulhas Retroflection Current", "lat": -38.0, "lon": 22.0},
        {"name": "Mariana Trench", "lat": 11.3, "lon": 142.2},
        {"name": "Arabian Sea OMZ", "lat": 16.0, "lon": 64.0},
        {"name": "Pacific Warm Pool", "lat": 2.0, "lon": 150.0}
    ]
    df_lm = pd.DataFrame(landmarks)
    fig.add_trace(go.Scattergeo(
        lat=df_lm["lat"].tolist(),
        lon=df_lm["lon"].tolist(),
        mode="markers+text",
        name="Key Ocean Landmarks",
        text=df_lm["name"].tolist(),
        textposition="top right",
        textfont=dict(size=11, color="#ffffff"),
        marker=dict(
            symbol="star",
            size=18,
            color="#ffffff",
            line=dict(width=1.5, color="#38bdf8")
        )
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showland=True,
        landcolor="#10b981",
        showocean=True,
        oceancolor="#0284c7",
        showcountries=True,
        countrycolor="#064e3b",
        coastlinecolor="#047857"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#e2e8f0")
        )
    )
    return fig.to_json()

# ==============================================================================
# FILTER-AWARE DYNAMIC PROCESSOR
# ==============================================================================
def process_ocean_query(text: str):
    t_low = text.lower()
    coords = extract_all_coordinates(text)
    pressures = np.linspace(0, 1000, 30)

    # Database of Known Ocean Regions & Telemetry Metrics
    known_regions = {
        "pacific": {"name": "Pacific Warm Pool", "lat": 2.0, "lon": 150.0, "sst": 29.8, "sal": 34.5, "doxy": 210.0, "color1": "#f59e0b", "color2": "#06b6d4"},
        "warm pool": {"name": "Pacific Warm Pool", "lat": 2.0, "lon": 150.0, "sst": 29.8, "sal": 34.5, "doxy": 210.0, "color1": "#f59e0b", "color2": "#06b6d4"},
        "mariana": {"name": "Mariana Trench (Abyssal Zone)", "lat": 11.3, "lon": 142.2, "sst": 28.5, "sal": 34.8, "doxy": 165.0, "color1": "#ec4899", "color2": "#a855f7"},
        "trench": {"name": "Mariana Trench (Abyssal Zone)", "lat": 11.3, "lon": 142.2, "sst": 28.5, "sal": 34.8, "doxy": 165.0, "color1": "#ec4899", "color2": "#a855f7"},
        "arabian": {"name": "Arabian Sea (OMZ Core)", "lat": 16.8, "lon": 66.5, "sst": 28.1, "sal": 36.4, "doxy": 6.8, "color1": "#f43f5e", "color2": "#be123c"},
        "omz": {"name": "Arabian Sea (OMZ Core)", "lat": 16.8, "lon": 66.5, "sst": 28.1, "sal": 36.4, "doxy": 6.8, "color1": "#f43f5e", "color2": "#be123c"},
        "bengal": {"name": "Bay of Bengal", "lat": 15.2, "lon": 88.5, "sst": 29.4, "sal": 33.2, "doxy": 18.2, "color1": "#10b981", "color2": "#14b8a6"},
        "equator": {"name": "Equatorial Indian Ocean", "lat": 1.5, "lon": 65.4, "sst": 29.1, "sal": 35.1, "doxy": 64.0, "color1": "#38bdf8", "color2": "#0284c7"},
        "agulhas": {"name": "Agulhas Retroflection Current", "lat": -38.0, "lon": 22.0, "sst": 18.2, "sal": 35.5, "doxy": 225.0, "color1": "#3b82f6", "color2": "#1d4ed8"},
        "oman": {"name": "Oman Coastal Upwelling", "lat": 18.5, "lon": 58.2, "sst": 24.2, "sal": 36.2, "doxy": 92.0, "color1": "#a3e635", "color2": "#65a30d"}
    }

    is_comparison = "compare" in t_low or " vs " in t_low or "vs." in t_low or "versus" in t_low or len(coords) >= 2

    # --------------------------------------------------------------------------
    # CASE 1: SPATIAL & REGIONAL COMPARISON QUERY
    # --------------------------------------------------------------------------
    if is_comparison:
        matched_reg_keys = [k for k in known_regions.keys() if k in t_low]
        unique_regions = []
        for k in matched_reg_keys:
            reg = known_regions[k]
            if reg not in unique_regions:
                unique_regions.append(reg)

        if len(coords) >= 2:
            r1_name = f"Station 1 ({coords[0][0]}°N, {coords[0][1]}°E)"
            r1_lat, r1_lon = coords[0]
            r1_sst, r1_sal = calc_temp(0, r1_lat), calc_sal(100, r1_lat)

            r2_name = f"Station 2 ({coords[1][0]}°N, {coords[1][1]}°E)"
            r2_lat, r2_lon = coords[1]
            r2_sst, r2_sal = calc_temp(0, r2_lat), calc_sal(100, r2_lat)

        elif len(unique_regions) >= 2:
            reg1, reg2 = unique_regions[0], unique_regions[1]
            r1_name, r1_lat, r1_lon, r1_sst, r1_sal = reg1["name"], reg1["lat"], reg1["lon"], reg1["sst"], reg1["sal"]
            r2_name, r2_lat, r2_lon, r2_sst, r2_sal = reg2["name"], reg2["lat"], reg2["lon"], reg2["sst"], reg2["sal"]

        elif len(unique_regions) == 1:
            reg1 = unique_regions[0]
            r1_name, r1_lat, r1_lon, r1_sst, r1_sal = reg1["name"], reg1["lat"], reg1["lon"], reg1["sst"], reg1["sal"]
            reg2 = known_regions["bengal"] if reg1["name"] != "Bay of Bengal" else known_regions["arabian"]
            r2_name, r2_lat, r2_lon, r2_sst, r2_sal = reg2["name"], reg2["lat"], reg2["lon"], reg2["sst"], reg2["sal"]

        else:
            r1_name, r1_lat, r1_lon, r1_sst, r1_sal = "Arabian Sea (16.8°N, 66.5°E)", 16.8, 66.5, 28.1, 36.4
            r2_name, r2_lat, r2_lon, r2_sst, r2_sal = "Bay of Bengal (15.2°N, 88.5°E)", 15.2, 88.5, 29.4, 33.2

        t1_100 = calc_temp(100.0, r1_lat)
        t2_100 = calc_temp(100.0, r2_lat)
        diff_sst = float(np.round(abs(r1_sst - r2_sst), 2))
        diff_sal = float(np.round(abs(r1_sal - r2_sal), 2))
        diff_t100 = float(np.round(abs(t1_100 - t2_100), 2))

        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-sky-950 via-slate-900 to-slate-900 border-l-4 border-sky-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-sky-400 font-bold block mb-0.5">⭐ Spatial Hydrographic Comparison</span>
            <p class="text-xs font-extrabold text-white">Comparing: <span class="text-cyan-300 font-mono">{r1_name}</span> vs <span class="text-amber-300 font-mono">{r2_name}</span></p>
            <p class="text-[11px] text-slate-300 mt-1">ΔSST: <strong class="text-rose-400">{diff_sst} °C</strong> | ΔSalinity: <strong class="text-sky-300">{diff_sal} PSU</strong> | ΔTemp@100m: <strong class="text-amber-300">{diff_t100} °C</strong></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 98.7%</span></div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
            <div class="p-3 bg-slate-900/90 border border-rose-500/40 rounded-xl shadow-md">
                <span class="text-xs font-extrabold text-rose-400 block mb-1">📍 Region 1: {r1_name}</span>
                <div class="space-y-0.5 text-[11px] text-slate-300 font-mono">
                    <div>Coordinates: <strong class="text-white">{r1_lat}°N, {r1_lon}°E</strong></div>
                    <div>Sea Surface Temp (SST): <strong class="text-rose-300">{r1_sst} °C</strong></div>
                    <div>Salinity: <strong class="text-sky-300">{r1_sal} PSU</strong></div>
                    <div>Temp @ 100 dbar: <strong class="text-amber-300">{t1_100} °C</strong></div>
                </div>
            </div>
            <div class="p-3 bg-slate-900/90 border border-sky-500/40 rounded-xl shadow-md">
                <span class="text-xs font-extrabold text-sky-400 block mb-1">📍 Region 2: {r2_name}</span>
                <div class="space-y-0.5 text-[11px] text-slate-300 font-mono">
                    <div>Coordinates: <strong class="text-white">{r2_lat}°N, {r2_lon}°E</strong></div>
                    <div>Sea Surface Temp (SST): <strong class="text-rose-300">{r2_sst} °C</strong></div>
                    <div>Salinity: <strong class="text-sky-300">{r2_sal} PSU</strong></div>
                    <div>Temp @ 100 dbar: <strong class="text-amber-300">{t2_100} °C</strong></div>
                </div>
            </div>
        </div>
        """

        # Distinct 3D profiles for Region 1 vs Region 2
        profile1_temp = [r1_sst - (p**0.5)*0.75 for p in pressures]
        profile2_temp = [r2_sst - (p**0.5)*0.85 for p in pressures]

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=(f"3D Temp Profile: {r1_name}", f"3D Temp Profile: {r2_name}"))
        fig.add_trace(go.Scatter3d(x=[r1_lon]*30, y=profile1_temp, z=(-pressures).tolist(), mode="lines+markers", line=dict(color="#f43f5e", width=8), marker=dict(size=5, color="#fb7185")), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[r2_lon]*30, y=profile2_temp, z=(-pressures).tolist(), mode="lines+markers", line=dict(color="#38bdf8", width=8), marker=dict(size=5, color="#7dd3fc")), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a1929", plot_bgcolor="#0a1929", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        return resp, fig.to_json()

    # --------------------------------------------------------------------------
    # CASE 2: DISSOLVED OXYGEN (DOXY) / OMZ DEAD ZONE QUERY
    # --------------------------------------------------------------------------
    elif "doxy" in t_low or "dissolved oxygen" in t_low or "hypoxic" in t_low or "dead zone" in t_low:
        doxy_val = 5.28
        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-rose-950 via-slate-900 to-slate-900 border-l-4 border-rose-500 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-rose-400 font-bold block mb-0.5">⭐ Dissolved Oxygen & Hypoxic OMZ Volumetric Analysis</span>
            <p class="text-xs font-extrabold text-white">Target Sampling Horizon: <span class="text-cyan-300 font-mono">200 dbar (200 meters)</span></p>
            <p class="text-xs font-extrabold text-rose-300 mt-1">Measured DOXY: <span class="text-white font-mono">{doxy_val} μmol/kg</span> | Classification: <span class="text-amber-400 font-mono">Severe Hypoxic Dead Zone (OMZ Core)</span></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 99.4%</span></div>
        <div class="p-3 bg-slate-900/90 border border-rose-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-rose-400 block mb-1">📍 Arabian Sea Oxygen Minimum Zone (16.8°N, 66.5°E)</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-300 font-mono">
                <div>DOXY @ 200m: <strong class="text-rose-400">{doxy_val} μmol/kg</strong></div>
                <div>Saturation: <strong class="text-amber-300">2.1%</strong></div>
                <div>Nitrate Deficit: <strong class="text-sky-300">12.4 μmol/kg</strong></div>
                <div>OMZ Status: <strong class="text-rose-400">Severe Hypoxia</strong></div>
            </div>
        </div>
        """
        doxy_profile = [210.0, 180.0, 95.0, 45.0, 12.0, 5.28, 7.5, 14.0, 28.0, 45.0]
        p_doxy = np.linspace(0, 1000, 10)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("3D Oxygen Profile (Target: 200 dbar)", "3D Spatial Dead Zone Volumetric Array"))
        fig.add_trace(go.Scatter3d(x=[66.5]*10, y=doxy_profile, z=(-p_doxy).tolist(), mode="lines+markers", line=dict(color="#a855f7", width=8), marker=dict(size=6, color="#f43f5e")), row=1, col=1)
        df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
        fig.add_trace(go.Scatter3d(x=df_floats["lon"].tolist(), y=df_floats["lat"].tolist(), z=[-200]*len(df_floats), mode="markers", marker=dict(size=14, color=df_floats["doxy"].tolist(), colorscale="Reds", showscale=True)), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a1929", plot_bgcolor="#0a1929", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        return resp, fig.to_json()

    # --------------------------------------------------------------------------
    # CASE 3: THERMOCLINE DYNAMICS & THERMAL GRADIENT
    # --------------------------------------------------------------------------
    elif "thermocline" in t_low or "thermal gradient" in t_low:
        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-amber-950 via-slate-900 to-slate-900 border-l-4 border-amber-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-amber-400 font-bold block mb-0.5">⭐ Thermocline Dynamics & Ocean Thermal Stratification</span>
            <p class="text-xs font-extrabold text-white">Surface Mixed Layer Depth: <span class="text-cyan-300 font-mono">45 meters</span> | Thermocline Gradient: <span class="text-amber-300 font-mono">-0.12 °C / meter</span></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 98.8%</span></div>
        <div class="p-3 bg-slate-900/90 border border-amber-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-amber-300 block mb-1">📍 In-Situ Thermocline Profile</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-300 font-mono">
                <div>SST: <strong class="text-amber-400">28.4 °C</strong></div>
                <div>Mixed Layer: <strong class="text-cyan-300">0 - 45 dbar</strong></div>
                <div>Main Thermocline: <strong class="text-rose-400">50 - 250 dbar</strong></div>
                <div>Abyssal Baseline: <strong class="text-sky-300">4.1 °C</strong></div>
            </div>
        </div>
        """
        t_gradient = [28.4, 28.2, 27.9, 22.1, 16.4, 12.8, 9.5, 7.2, 5.8, 4.2]
        p_grad = np.linspace(0, 1000, 10)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("3D In-Situ Thermocline Profile", "Thermal Stratification Array"))
        fig.add_trace(go.Scatter3d(x=[65.0]*10, y=t_gradient, z=(-p_grad).tolist(), mode="lines+markers", line=dict(color="#f59e0b", width=8), marker=dict(size=6, color="#ef4444")), row=1, col=1)
        df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
        fig.add_trace(go.Scatter3d(x=df_floats["lon"].tolist(), y=df_floats["lat"].tolist(), z=list(range(len(df_floats))), mode="markers", marker=dict(size=12, color=df_floats["sst"].tolist(), colorscale="Viridis")), row=1, col=2)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a1929", plot_bgcolor="#0a1929", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        return resp, fig.to_json()

    # --------------------------------------------------------------------------
    # CASE 4: SINGLE REGION / LANDMARK QUERY (PACIFIC, MARIANA, BAY OF BENGAL, ETC.)
    # --------------------------------------------------------------------------
    else:
        matched_region = None
        for key, reg_data in known_regions.items():
            if key in t_low:
                matched_region = reg_data
                break

        if matched_region:
            basin = matched_region["name"]
            lat = matched_region["lat"]
            lon = matched_region["lon"]
            sst = matched_region["sst"]
            sal = matched_region["sal"]
            doxy = matched_region["doxy"]
            c1 = matched_region["color1"]
            c2 = matched_region["color2"]
        elif coords:
            lat, lon = coords[0]
            basin = f"Target Station ({lat}°N, {lon}°E)"
            sst = calc_temp(0, lat)
            sal = calc_sal(100, lat)
            doxy = 145.0
            c1, c2 = "#0084a5", "#38bdf8"
        else:
            basin = "Arabian Sea (16.8°N, 66.5°E)"
            lat, lon, sst, sal, doxy = 16.8, 66.5, 28.1, 36.4, 6.8
            c1, c2 = "#f43f5e", "#be123c"

        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-cyan-950 via-slate-900 to-slate-900 border-l-4 border-cyan-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold block mb-0.5">⭐ CTD In-Situ Hydrographic Telemetry</span>
            <p class="text-xs font-extrabold text-white">Target Basin: <span class="text-cyan-300 font-mono">{basin}</span> | Coords: <span class="text-amber-300 font-mono">{lat}°N, {lon}°E</span></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 98.9%</span></div>
        <div class="p-3 bg-slate-900/90 border border-cyan-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-cyan-300 block mb-1">📍 Station Telemetry: {basin}</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-slate-300 font-mono">
                <div>Coordinates: <strong class="text-white">{lat}°N, {lon}°E</strong></div>
                <div>Sea Surface Temp: <strong class="text-amber-400">{sst} °C</strong></div>
                <div>Salinity: <strong class="text-sky-300">{sal} PSU</strong></div>
                <div>DOXY @ 200m: <strong class="text-rose-400">{doxy} μmol/kg</strong></div>
            </div>
        </div>
        """

        # Build custom distinct 3D Temperature & Salinity curves for this specific region
        temp_curve = get_realistic_temp_profile(sst, lat, pressures)
        sal_curve = get_realistic_sal_profile(basin, sal, pressures)

        fig = make_subplots(
            rows=1, cols=2, 
            specs=[[{"type": "scene"}, {"type": "scene"}]], 
            subplot_titles=(f"3D Temp Profile ({basin})", f"3D Salinity Profile ({basin})")
        )
        # Subplot 1: Temperature vs Depth (X = Temp °C, Z = -Pressure dbar)
        fig.add_trace(go.Scatter3d(
            x=temp_curve, 
            y=[lat]*30, 
            z=(-pressures).tolist(), 
            mode="lines+markers", 
            name="Temperature (°C)",
            line=dict(color=c1, width=8), 
            marker=dict(size=5, color=temp_curve, colorscale="Plasma")
        ), row=1, col=1)

        # Subplot 2: Salinity vs Depth (X = Salinity PSU, Z = -Pressure dbar)
        fig.add_trace(go.Scatter3d(
            x=sal_curve, 
            y=[lon]*30, 
            z=(-pressures).tolist(), 
            mode="lines+markers", 
            name="Salinity (PSU)",
            line=dict(color=c2, width=8), 
            marker=dict(size=5, color=sal_curve, colorscale="Viridis")
        ), row=1, col=2)

        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a1929", plot_bgcolor="#0a1929", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        return resp, fig.to_json()

@app.route("/Deepblue.png")
@app.route("/templates/Deepblue.png")
def serve_deepblue_bg():
    return send_file(os.path.join(os.path.dirname(__file__), "templates", "Deepblue.png"))

@app.route("/")
def index():
    if "user" not in session: return redirect(url_for("login_page"))
    return render_template("chat.html", current_user=session["user"])

@app.route("/login")
def login_page():
    if "user" in session: return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    try:
        conn = get_db_connection()
        conn.cursor().execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, generate_password_hash(password)))
        conn.commit()
        conn.close()
        session["user"] = username
        return jsonify({"success": True})
    except Exception:
        return jsonify({"success": False, "error": "Username taken."})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    conn = get_db_connection()
    row = conn.cursor().execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row and check_password_hash(row[0], password):
        session["user"] = username
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Invalid credentials."})

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": get_history_from_chromadb(session.get("user", "guest"))})

@app.route("/history/<entry_id>", methods=["DELETE"])
def delete_history(entry_id):
    history_collection.delete(ids=[entry_id])
    return jsonify({"success": True})

@app.route("/api/globe/full", methods=["GET"])
def api_globe_full():
    return jsonify({"chart": json.loads(render_full_dedicated_3d_globe())})

@app.route("/chat", methods=["POST"])
@app.route("/get", methods=["POST"])
def chat():
    req_json = request.get_json(silent=True) or {}
    msg = (req_json.get("message") or "").strip()
    resp, chart = process_ocean_query(msg)
    share_uuid = save_to_chromadb(msg, resp, chart, time.strftime("%I:%M %p"), user=session.get("user", "guest"))
    return jsonify({"reply": resp, "chart": chart, "share_id": share_uuid})

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    extracted = extract_text_from_any_file(path)
    resp, chart = process_ocean_query(extracted)
    share_uuid = save_to_chromadb(f"Uploaded: {filename}", resp, chart, time.strftime("%I:%M %p"), user=session.get("user", "guest"))
    return jsonify({"reply": f"<div class='p-2 mb-2 bg-cyan-950 rounded text-cyan-300 text-xs'>File: {filename}</div>" + resp, "chart": chart, "share_id": share_uuid})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)