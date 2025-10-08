# app.py
# Versi 5.0: The Unified Agent (Backend + Frontend)

from fastapi import FastAPI, Request
from modal import Image, Stub, asgi_app, Secret, Volume
import sys
import os

# --- Konfigurasi Stub, Volume, dan Secret ---
stub = Stub(name="unified-architect-agent")
volume = Volume.persisted("intelligence-data-volume")
cerebras_secret = Secret.from_name("cerebras-api-key")

# --- Definisi Image Docker ---
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi", "uvicorn", "sentence-transformers", "faiss-cpu",
        "beautifulsoup4", "lxml", "cerebras-cloud-sdk", "numpy",
        "streamlit==1.33.0", # Pin versi streamlit untuk stabilitas
        "starlette-streamlit",
    )
    # Menambahkan direktori proyek ke PYTHONPATH
    .env({"PYTHONPATH": "/root"})
)

# --- Inisialisasi Aplikasi FastAPI ---
web_app = FastAPI(title="Unified Agent System v5.0")

# --- Fungsi Lifecycle & Endpoint Utama Modal ---
@stub.function(
    image=image,
    mounts=[
        # Mount semua file .py di direktori saat ini ke dalam container
        *modal.Mount.from_local_python_packages("agent", "ui")
    ],
    volumes={"/cache": volume},
    secrets=[cerebras_secret],
    container_idle_timeout=300,
    keep_warm=1,
    timeout=900 # Timeout lebih lama untuk misi yang kompleks
)
@asgi_app()
def unified_app():
    """
    Endpoint utama yang menyajikan FastAPI backend dan Streamlit frontend.
    """
    from fastapi import APIRouter
    from starlette_streamlit import StreamlitMiddleware
    
    from agent import IntelligenceSystem
    
    # 1. Siapkan Backend API
    agent_system = IntelligenceSystem()
    api_router = APIRouter(prefix="/api")

    # Wrapper untuk memanggil fungsi async dari FastAPI sync context
    @api_router.post("/execute_mission")
    async def execute(request: Request):
        mission_data = await request.json()
        return await agent_system.execute_mission(mission_data)

    web_app.include_router(api_router)
    
    # 2. Mount Streamlit UI ke aplikasi utama
    # Kita akan menjalankan file ui.py secara langsung
    web_app.add_middleware(StreamlitMiddleware, app_path="ui.py")
    
    return web_app
