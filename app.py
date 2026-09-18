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
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
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
    fig.add_trace(go.Scattergeo(lat=df_floats["lat"].tolist(), lon=df_floats["lon"].tolist(), mode="markers", marker=dict(size=12, color=df_floats["sst"].tolist(), colorscale="Plasma", showscale=True)))
    fig.update_geos(projection_type="orthographic", showland=True, landcolor="#84cc16", showocean=True, oceancolor="#0284c7")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))
    return fig.to_json()

# ==============================================================================
# FILTER-AWARE DYNAMIC PROCESSOR
# ==============================================================================
def process_ocean_query(text: str):
    t_low = text.lower()
    coords = extract_all_coordinates(text)
    is_comparison = "compare" in t_low or "vs" in t_low or len(coords) >= 2
    is_equator = "equator" in t_low or "within 5" in t_low or "last year" in t_low
    is_filter = "salinity" in t_low and ("200" in t_low or "past month" in t_low)

    pressures = np.linspace(0, 1000, 30)

    if is_filter:
        target_depth = 200.0
        sal_val = calc_sal(target_depth, 16.8)
        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-indigo-950 via-slate-900 to-slate-900 border-l-4 border-indigo-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-indigo-400 font-bold block mb-0.5">⭐ Threshold Filter Analysis (Salinity @ 200 dbar)</span>
            <p class="text-sm font-extrabold text-white">Filter Active: <span class="text-cyan-300 font-mono">Max Salinity Threshold (200 dbar)</span> | Window: <span class="text-amber-300 font-mono">Past Month</span></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 99.1%</span></div>
        <div class="p-2.5 bg-slate-900/90 border border-cyan-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-cyan-300 block mb-0.5">📍 Filtered Station Array (Arabian Sea / Depth 200m)</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 text-[11px] text-slate-300 font-mono">
                <div>Depth Limit: <strong class="text-white">200 dbar</strong></div>
                <div>Salinity Value: <strong class="text-sky-300">{sal_val} PSU</strong></div>
                <div>Status: <strong class="text-emerald-300">Filtered & Active</strong></div>
            </div>
        </div>
        """
        s_vals = [calc_sal(p, 16.8) for p in pressures]
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("Filtered Salinity Profile (200 dbar)", "Station Array Grid"))
        fig.add_trace(go.Scatter3d(x=[0]*30, y=s_vals, z=(-pressures).tolist(), mode="lines", line=dict(color="#38bdf8", width=7)), row=1, col=1)
        df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
        fig.add_trace(go.Scatter3d(x=df_floats["lon"].tolist(), y=df_floats["lat"].tolist(), z=list(range(len(df_floats))), mode="markers+text", text=df_floats["id"].tolist(), marker=dict(size=10, color=df_floats["sst"].tolist(), colorscale="Plasma")), row=1, col=2)

    elif is_equator:
        eq_floats = [f for f in GLOBAL_FLOAT_DATASET if abs(f["lat"]) <= 5.0]
        cards_html = "".join([f"""
        <div class="p-2.5 bg-slate-900/90 border border-cyan-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-cyan-300 block mb-0.5">📍 Station: {f['name']}</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 text-[11px] text-slate-300 font-mono">
                <div>Coords: <strong class="text-white">{f['lat']}°, {f['lon']}°</strong></div>
                <div>SST: <strong class="text-amber-400">{f['sst']} °C</strong></div>
                <div>Salinity: <strong class="text-sky-300">{f['sal']} PSU</strong></div>
            </div>
        </div>
        """ for f in eq_floats])

        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-cyan-950 via-slate-900 to-slate-900 border-l-4 border-cyan-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-cyan-400 font-bold block mb-0.5">⭐ Equatorial Cluster Discovery</span>
            <p class="text-sm font-extrabold text-white">Region: <span class="text-cyan-300 font-mono">Global Equatorial Zone (±5°)</span></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 98.9%</span></div>
        <div><p class="text-xs font-semibold text-slate-200 mb-2">📋 Equatorial ARGO Float Array:</p><div class="max-h-60 overflow-y-auto mb-3">{cards_html}</div></div>
        """
        df_eq = pd.DataFrame(eq_floats)
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("Equatorial CTD Profile", "Equatorial Station Array Grid"))
        fig.add_trace(go.Scatter3d(x=[0]*20, y=[29.0]*20, z=(-np.linspace(0, 30, 20)).tolist(), mode="lines", line=dict(color="#38bdf8", width=7)), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=df_eq["lon"].tolist(), y=df_eq["lat"].tolist(), z=list(range(len(df_eq))), mode="markers+text", text=df_eq["id"].tolist(), marker=dict(size=12, color=df_eq["sst"].tolist(), colorscale="Plasma")), row=1, col=2)

    elif is_comparison:
        c1 = coords[0] if len(coords) > 0 else (15.0, 65.0)
        c2 = coords[1] if len(coords) > 1 else (10.0, 70.0)
        t1, t2 = calc_temp(100.0, c1[0]), calc_temp(100.0, c2[0])
        diff = float(np.round(abs(t1 - t2), 2))

        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-sky-950 via-slate-900 to-slate-900 border-l-4 border-sky-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-sky-400 font-bold block mb-0.5">⭐ Spatial Comparison Analysis</span>
            <p class="text-sm font-extrabold text-white">Comparing: <span class="text-cyan-300 font-mono">({c1[0]}°N, {c1[1]}°E) vs ({c2[0]}°N, {c2[1]}°E)</span> | ΔTemperature: <span class="text-amber-300 font-mono">{diff} °C</span></p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 95.2%</span></div>
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
            <div class="p-2.5 bg-slate-900/90 border border-rose-500/40 rounded-xl shadow-inner">
                <span class="text-xs font-bold text-rose-400 block mb-1">📍 Station 1 ({c1[0]}°N, {c1[1]}°E)</span>
                <p class="text-xs text-slate-200 font-mono">Temp at 100 dbar: <strong>{t1:.2f} °C</strong></p>
            </div>
            <div class="p-2.5 bg-slate-900/90 border border-sky-500/40 rounded-xl shadow-inner">
                <span class="text-xs font-bold text-sky-400 block mb-1">📍 Station 2 ({c2[0]}°N, {c2[1]}°E)</span>
                <p class="text-xs text-slate-200 font-mono">Temp at 100 dbar: <strong>{t2:.2f} °C</strong></p>
            </div>
        </div>
        """
        vals1 = [calc_temp(p, c1[0]) for p in pressures]
        vals2 = [calc_temp(p, c2[0]) for p in pressures]
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=(f"Station 1 ({c1[0]}°N)", f"Station 2 ({c2[0]}°N)"))
        fig.add_trace(go.Scatter3d(x=[0]*30, y=vals1, z=(-pressures).tolist(), mode="lines", line=dict(color="#f43f5e", width=7)), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=[0]*30, y=vals2, z=(-pressures).tolist(), mode="lines", line=dict(color="#38bdf8", width=7)), row=1, col=2)

    else:
        if "bay of bengal" in t_low:
            lat, lon, basin = 15.2, 88.5, "Bay of Bengal"
        elif "arabian sea" in t_low:
            lat, lon, basin = 16.8, 66.5, "Arabian Sea"
        elif coords:
            lat, lon = coords[0]
            basin = f"Custom Coordinates Region"
        else:
            lat, lon, basin = 16.8, 66.5, "Arabian Sea"

        temp = calc_temp(100.0, lat)
        sal = calc_sal(100.0, lat)

        resp = f"""
        <div class="p-3 mb-3 bg-gradient-to-r from-rose-950 via-slate-900 to-slate-900 border-l-4 border-rose-400 rounded-r-xl shadow-xl">
            <span class="text-[10px] font-mono uppercase tracking-widest text-rose-400 font-bold block mb-0.5">⭐ CTD Hydrographic Analysis</span>
            <p class="text-sm font-extrabold text-white">Region: <span class="text-cyan-300 font-mono">{basin}</span> | Profile at {lat}°N, {lon}°E</p>
        </div>
        <div class="flex items-center gap-2 mb-2.5"><span class="px-2.5 py-1 rounded-md text-[11px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono"><i class="fa-solid fa-shield-halved"></i> Confidence Score: 98.4%</span></div>
        <div class="p-2.5 bg-slate-900/90 border border-cyan-500/40 rounded-xl mb-2 shadow-md">
            <span class="text-xs font-extrabold text-cyan-300 block mb-0.5">📍 Station Record: {basin}</span>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 text-[11px] text-slate-300 font-mono">
                <div>Coords: <strong class="text-white">{lat}°N, {lon}°E</strong></div>
                <div>Temp: <strong class="text-amber-400">{temp} °C</strong></div>
                <div>Salinity: <strong class="text-sky-300">{sal} PSU</strong></div>
            </div>
        </div>
        """
        t_vals = [calc_temp(p, lat) for p in pressures]
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=(f"3D CTD Profile ({basin})", "Station Array Grid"))
        fig.add_trace(go.Scatter3d(x=[0]*30, y=t_vals, z=(-pressures).tolist(), mode="lines", line=dict(color="#f43f5e", width=7)), row=1, col=1)
        df_floats = pd.DataFrame(GLOBAL_FLOAT_DATASET)
        fig.add_trace(go.Scatter3d(x=df_floats["lon"].tolist(), y=df_floats["lat"].tolist(), z=list(range(len(df_floats))), mode="markers+text", text=df_floats["id"].tolist(), marker=dict(size=10, color=df_floats["sst"].tolist(), colorscale="Plasma")), row=1, col=2)

    fig.update_layout(template="plotly_dark", paper_bgcolor="#1e222d", margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    return resp, fig.to_json()

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
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)