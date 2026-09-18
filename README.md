# 🌊 FloatChat AI — Oceanographic Telemetry & ARGO Intelligence Portal

![FloatChat AI Banner](https://img.shields.io/badge/FloatChat%20AI-v2.0%20Oceanographic%20Portal-0084a5?style=for-the-badge&logo=waves&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly.js-3D%20Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Storage-orange?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite3-User%20Auth-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-Glassmorphic%20UI-38BDF8?style=for-the-badge&logo=tailwindcss&logoColor=white)

**FloatChat AI** is an advanced oceanographic intelligence portal designed for real-time **ARGO Float Data Discovery, 3D Geographic Exploration, and Physics-Based CTD (Conductivity, Temperature, Depth) Visualization**.

Featuring a sleek, glassmorphic UI set against a deep-sea coral reef backdrop (`Deepblue.png`), FloatChat AI empowers oceanographers, climate researchers, and data analysts to interactively query, visualize, and compare global oceanic telemetry.

---

## 🌟 Key Features

### 💬 1. Multimodal AI Chat Workspace
- **Context-Aware Query Engine**: Ask complex questions about sea surface temperatures, salinity anomalies, oxygen minimum zones (OMZ), thermocline gradients, and current ocean dynamics.
- **Side-by-Side Spatial Comparisons**: Easily compare distinct marine regions (e.g., *Pacific Warm Pool vs Mariana Trench*, *Arabian Sea vs Bay of Bengal*).
- **Interactive 3D Subplot Visualizers**: Automatic rendering of dual 3D curve profiles (Temperature vs Depth & Salinity vs Depth) using Plotly.js.
- **Multilingual Speech Dictation**: Hands-free voice input supporting English (`en`), Hindi (`hi`), and Marathi (`mr`).
- **File Attachment Parser**: Upload PDF research papers, CSV datasets, or TXT telemetry logs directly into the AI workspace for instant extraction and analysis.
- **Chart Exports**: One-click download of interactive 3D visualizations in high-res PNG or SVG vector formats.

### 🌐 2. 3D Globe Explorer
- **Interactive Orthographic 3D Globe**: Visual rendering of global ARGO float stations, landmark oceanic trenches, and key hydrographic research sites.
- **Real-Time Station Inspection**: Click on any ARGO float station or landmark star to inspect live latitude, longitude, ocean basin, and depth ratings.
- **One-Click Query Redirection**: Click **"Query in Chat ➔"** on any station info card to instantly transfer location metrics into the AI Workspace for deep analysis.
- **Dynamic Camera Controls**: Reset view, toggle auto-rotation, or focus on specific ocean basins seamlessly.

### 🔐 3. Authentication & Persistent Vector History
- **SQLite User Authentication**: Secure password hashing (`werkzeug.security`) with 30-day session cookie persistence.
- **ChromaDB Vector History**: Store, recall, and search past chat sessions with vector similarity search. Delete or reload history entries instantly.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Backend Framework** | **Python Flask** | RESTful routing, session management, file upload handling, and telemetry endpoints |
| **Vector Storage** | **ChromaDB** | Persistent vector database for context history & similarity search |
| **Relational Database**| **SQLite 3 (WAL mode)**| User authentication schema with encrypted credentials |
| **Data Processing** | **Pandas, NumPy, PyPDF** | CTD profile physics calculations, CSV parsing, and PDF text extraction |
| **Data Visualization**| **Plotly.js** | Interactive 3D orthographic globe & 3D CTD depth subplots |
| **Frontend Styling** | **Tailwind CSS + Glassmorphism**| Custom responsive glass layout with `Deepblue.png` coral backdrop |
| **Voice Dictation** | **Web Speech API** | Multilingual speech-to-text recognition |

---

## 🚀 Quick Start Guide

### Prerequisites
- Python **3.9+** installed
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Pradnyadange/Float-Chat.git
cd Float-Chat
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, run: `pip install flask pandas numpy plotly chromadb pypdf werkzeug python-dotenv`)*

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Portal
Open your web browser and navigate to:
```
http://127.0.0.1:8080
```

---

## 📁 Repository Structure

```
Float-Chat/
├── app.py                   # Main Flask backend application & query logic
├── floatchat_users.db       # SQLite user authentication database (auto-generated)
├── chroma_db/               # ChromaDB vector store directory (auto-generated)
├── uploads/                 # Uploaded research files directory
├── templates/
│   ├── chat.html            # AI Chat Workspace & 3D Globe Explorer UI
│   ├── login.html           # Glassmorphic Login & Register portal
│   └── Deepblue.png         # High-resolution ocean background asset
├── README.md                # Project documentation
└── .gitignore               # Git ignore patterns
```

---

## 📊 Example Queries to Try

- `"Show me the temperature profile of the Mariana Trench"`
- `"Compare the Pacific Warm Pool and the Arabian Sea"`
- `"What is the Oxygen Minimum Zone (OMZ) depth in the Arabian Sea?"`
- `"Analyze thermocline dynamics in the Agulhas Current"`
- `"How does salinity vary with depth in the Bay of Bengal?"`

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---

<p center="align">
  <b>FloatChat AI</b> — Empowering Oceanographic Research & Climate Telemetry.
</p>