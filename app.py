# app.py (Versi FINAL: FastAPI + Native HTML/JS UI)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import modal
import os
import sys

# Tambahkan direktori proyek agar bisa mengimpor file lain
sys.path.append(os.path.dirname(__file__))

# --- Konfigurasi Stub, Volume, dan Secret ---
stub = modal.App(name="simple-web-agent")
volume = modal.Volume.persisted("intelligence-data-volume")
cerebras_secret = modal.Secret.from_name("cerebras-api-key")

# --- Definisi Image Docker (DIHAPUS: Streamlit dan starlette-streamlit) ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi", "uvicorn", "sentence-transformers", "faiss-cpu",
        "beautifulsoup4", "lxml", "cerebras-cloud-sdk", "numpy",
    )
    .env({"PYTHONPATH": "/root"})
)

# --- Kelas Agent Logic diimpor dari agent.py ---
from agent import IntelligenceSystem

# --- Inisialisasi Aplikasi FastAPI ---
web_app = FastAPI(title="Simple Agent System")


# --- UI: Halaman Utama (FastAPI melayani HTML) ---
@web_app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """
    Endpoint utama yang melayani frontend HTML/JavaScript statis.
    """
    
    # URL API akan sama dengan URL root aplikasi, diikuti oleh /api/execute_mission
    api_url = request.url_for("execute_mission")

    # Ini adalah HTML/JS/CSS minimal yang canggih
    html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Architect Agent</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; display: flex; justify-content: center; }}
        .container {{ width: 100%; max-width: 800px; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }}
        h1 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }}
        textarea {{ width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; resize: vertical; }}
        button {{ background-color: #0056b3; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }}
        button:hover {{ background-color: #004085; }}
        #output {{ margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; white-space: pre-wrap; }}
        .loading-spinner {{ display: none; margin: 20px auto; border: 4px solid rgba(0, 0, 0, 0.1); border-top: 4px solid #0056b3; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Architect Agent: Autonomous System</h1>
        <p>Masukkan misi tingkat tinggi Anda. Agen akan merencanakan, mengeksekusi, dan menyajikan laporan akhir.</p>
        
        <textarea id="missionInput" rows="5" placeholder="Contoh: Selidiki aktivitas terbaru grup ransomware 'LockBit'." required></textarea>
        
        <button onclick="executeMission()">🚀 Jalankan Misi</button>
        
        <div id="loadingSpinner" class="loading-spinner"></div>
        
        <div id="output">
            <strong>Laporan Akhir:</strong>
            <p>Hasil analisis akan muncul di sini...</p>
        </div>
    </div>

    <script>
        const API_URL = "{api_url}";

        async function executeMission() {{
            const mission = document.getElementById('missionInput').value;
            const outputDiv = document.getElementById('output');
            const spinner = document.getElementById('loadingSpinner');

            if (!mission) {{
                alert("Misi tidak boleh kosong!");
                return;
            }}

            outputDiv.innerHTML = '<strong>Laporan Akhir:</strong><p>Agen sedang bekerja...</p>';
            spinner.style.display = 'block';

            try {{
                const response = await fetch(API_URL, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ mission: mission }})
                }});

                if (!response.ok) {{
                    throw new Error(`Error HTTP: ${response.status}`);
                }}

                const data = await response.json();
                
                // Menampilkan Log Eksekusi
                let logHtml = '<h4>Log Eksekusi Rencana:</h4><pre>' + JSON.stringify(data.execution_log, null, 2) + '</pre>';

                // Menampilkan Laporan Akhir
                outputDiv.innerHTML = '<strong>Laporan Akhir (Final Report):</strong><br>' + data.final_report.replace(/\\n/g, '<br>') + '<hr>' + logHtml;


            }} catch (error) {{
                outputDiv.innerHTML = `<strong>ERROR:</strong> Terjadi kesalahan dalam pemanggilan API atau jaringan. ${error.message}`;
                console.error('Error:', error);
            }} finally {{
                spinner.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# --- API: Endpoint Logika (Backend) ---
@modal.function(
    image=image,
    volumes={"/cache": volume},
    secrets=[cerebras_secret],
    container_idle_timeout=300,
    keep_warm=1,
    timeout=900
)
@web_app.post("/api/execute_mission")
async def execute_mission(mission_data: dict):
    """
    Endpoint utama yang dipanggil oleh frontend untuk menjalankan logika agen.
    """
    # Import logic dari agent.py di sini untuk memastikan scope yang benar
    agent_system = IntelligenceSystem()
    return await agent_system.execute_mission(mission_data)


# --- Deployment Entry Point ---
@modal.asgi_app()
def final_app():
    return web_app
