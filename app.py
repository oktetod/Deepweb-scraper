"""
FastAPI Application - Fixed for Modal v0.63+
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import modal
import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(__file__))

# ============================================================================
# MODAL CONFIGURATION (v0.63+ Compatible)
# ============================================================================

app = modal.App(name="intelligence-agent-system")

# Persisted volume for data storage
volume = modal.Volume.from_name("intelligence-data-vol", create_if_missing=True)

# Secrets
cerebras_secret = modal.Secret.from_name("cerebras-api-key")

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "sentence-transformers==2.2.2",
        "faiss-cpu==1.7.4",
        "beautifulsoup4==4.12.2",
        "lxml==4.9.3",
        "cerebras-cloud-sdk==1.0.0",
        "numpy==1.24.3",
    )
    .env({"PYTHONPATH": "/root"})
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class MissionRequest(BaseModel):
    """Request model for mission execution"""
    mission: str = Field(..., min_length=10, max_length=2000)
    
    @field_validator('mission')
    @classmethod
    def validate_mission(cls, v):
        if not v.strip():
            raise ValueError('Mission cannot be empty')
        return v.strip()

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    timestamp: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    uptime: float
    documents: int
    cache_size: int
    timestamp: float

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

web_app = FastAPI(
    title="Intelligence Agent System",
    description="Autonomous AI agent for intelligence missions",
    version="2.0.0"
)

# Middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
web_app.add_middleware(GZipMiddleware, minimum_size=1000)

# Logging middleware
@web_app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"← {request.method} {request.url.path} [{response.status_code}] {duration:.3f}s")
    return response

# Security headers
@web_app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

# ============================================================================
# FRONTEND UI
# ============================================================================

@web_app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve modern web interface"""
    api_url = str(request.url_for("execute_mission"))
    health_url = str(request.url_for("health_check"))
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligence Agent System</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        :root {{
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --success: #10b981;
            --error: #ef4444;
            --bg: #0f172a;
            --surface: #1e293b;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --border: #334155;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #1e293b 100%);
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{ max-width: 1000px; margin: 0 auto; }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 20px;
            background: var(--surface);
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .status-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            border-radius: 10px;
            margin-bottom: 25px;
        }}
        
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
            margin-right: 10px;
            display: inline-block;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}
        
        .card {{
            background: var(--surface);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 25px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        textarea {{
            width: 100%;
            padding: 16px;
            background: var(--bg);
            border: 2px solid var(--border);
            border-radius: 10px;
            color: var(--text);
            font-size: 1rem;
            resize: vertical;
            min-height: 140px;
            transition: all 0.3s;
            font-family: inherit;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1);
        }}
        
        .button-group {{
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }}
        
        button {{
            flex: 1;
            padding: 16px 32px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        
        .btn-primary {{
            background: var(--primary);
            color: white;
        }}
        
        .btn-primary:hover:not(:disabled) {{
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
        }}
        
        .btn-primary:disabled {{
            background: var(--border);
            cursor: not-allowed;
            transform: none;
        }}
        
        .btn-secondary {{
            background: transparent;
            color: var(--text);
            border: 2px solid var(--border);
        }}
        
        .btn-secondary:hover {{
            border-color: var(--primary);
            background: rgba(59, 130, 246, 0.1);
        }}
        
        .spinner {{
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        
        .spinner.active {{ display: inline-block; }}
        
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        
        .output {{
            display: none;
            margin-top: 25px;
        }}
        
        .output.active {{ display: block; }}
        
        .report-header {{
            background: linear-gradient(135deg, var(--primary), var(--success));
            color: white;
            padding: 20px;
            border-radius: 10px 10px 0 0;
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .report-content {{
            background: var(--bg);
            padding: 25px;
            border-radius: 0 0 10px 10px;
            border: 1px solid var(--border);
            line-height: 1.8;
        }}
        
        .report-content h2 {{
            color: var(--primary);
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.5rem;
        }}
        
        .report-content h3 {{
            color: var(--success);
            margin-top: 20px;
            margin-bottom: 12px;
            font-size: 1.25rem;
        }}
        
        .report-content h4 {{
            color: var(--text);
            margin-top: 15px;
            margin-bottom: 10px;
        }}
        
        .report-content strong {{ color: var(--primary); }}
        .report-content em {{ color: var(--success); }}
        
        .execution-log {{
            margin-top: 20px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid var(--border);
        }}
        
        .execution-log::-webkit-scrollbar {{ width: 10px; }}
        .execution-log::-webkit-scrollbar-track {{ background: var(--bg); }}
        .execution-log::-webkit-scrollbar-thumb {{ 
            background: var(--border); 
            border-radius: 5px;
        }}
        
        .success-badge {{
            display: inline-block;
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid var(--success);
            color: var(--success);
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        .error-badge {{
            display: inline-block;
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid var(--error);
            color: var(--error);
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8rem; }}
            .button-group {{ flex-direction: column; }}
            .status-bar {{ flex-direction: column; gap: 10px; text-align: center; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Intelligence Agent System</h1>
            <p style="color: var(--text-dim); font-size: 1.1rem; margin-top: 10px;">
                Autonomous AI-powered intelligence analysis platform
            </p>
        </div>
        
        <div class="status-bar" id="statusBar">
            <div>
                <span class="status-dot"></span>
                <span id="statusText">System Online</span>
            </div>
            <span id="systemInfo">Loading...</span>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 10px; color: var(--primary);">Mission Control</h2>
            <p style="color: var(--text-dim); margin-bottom: 20px; line-height: 1.6;">
                Describe your intelligence objective. The AI agent will autonomously plan, 
                execute multi-step investigations, and deliver a comprehensive report.
            </p>
            
            <textarea 
                id="missionInput" 
                placeholder="Example: Investigate recent LockBit ransomware attacks and analyze their TTPs..."
            ></textarea>
            
            <div class="button-group">
                <button class="btn-primary" onclick="executeMission()" id="executeBtn">
                    <span id="btnText">🚀 Execute Mission</span>
                    <div class="spinner" id="spinner"></div>
                </button>
                <button class="btn-secondary" onclick="clearAll()">
                    🗑️ Clear
                </button>
            </div>
        </div>
        
        <div class="output" id="output">
            <div class="report-header">📊 Mission Report</div>
            <div class="report-content" id="reportContent"></div>
            <div class="execution-log" id="executionLog"></div>
        </div>
    </div>

    <script>
        const API_URL = "{api_url}";
        const HEALTH_URL = "{health_url}";
        
        async function checkHealth() {{
            try {{
                const res = await fetch(HEALTH_URL);
                const data = await res.json();
                document.getElementById('systemInfo').textContent = 
                    `📚 ${{data.documents}} docs | ⏱️ ${{Math.floor(data.uptime)}}s uptime`;
            }} catch (e) {{
                console.error('Health check failed:', e);
                document.getElementById('systemInfo').textContent = '⚠️ Health check failed';
            }}
        }}
        
        checkHealth();
        setInterval(checkHealth, 30000);
        
        async function executeMission() {{
            const mission = document.getElementById('missionInput').value.trim();
            const output = document.getElementById('output');
            const reportContent = document.getElementById('reportContent');
            const execLog = document.getElementById('executionLog');
            const btn = document.getElementById('executeBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');

            if (!mission || mission.length < 10) {{
                alert('⚠️ Mission must be at least 10 characters long');
                return;
            }}

            btn.disabled = true;
            btnText.textContent = 'Processing...';
            spinner.classList.add('active');
            output.classList.add('active');
            reportContent.innerHTML = '<p style="color: var(--text-dim); padding: 20px;">🔄 AI agent is planning and executing your mission...</p>';
            execLog.style.display = 'none';

            const startTime = Date.now();

            try {{
                const res = await fetch(API_URL, {{
                    method: 'POST',
                    headers: {{ 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }},
                    body: JSON.stringify({{ mission }})
                }});

                const duration = ((Date.now() - startTime) / 1000).toFixed(2);

                if (!res.ok) {{
                    const err = await res.json();
                    throw new Error(err.error || err.detail || `HTTP ${{res.status}}`);
                }}

                const data = await res.json();
                
                if (data.success) {{
                    reportContent.innerHTML = `
                        <div class="success-badge">
                            ✅ Mission completed successfully in ${{duration}}s
                        </div>
                        ${{formatMarkdown(data.final_report)}}
                    `;
                }} else {{
                    reportContent.innerHTML = `
                        <div class="error-badge">
                            ❌ Mission failed: ${{data.error || 'Unknown error'}}
                        </div>
                        <p style="color: var(--text-dim); margin-top: 15px;">
                            Please try again or refine your mission description.
                        </p>
                    `;
                }}
                
                if (data.execution_log && data.execution_log.length > 0) {{
                    execLog.style.display = 'block';
                    execLog.innerHTML = '<strong style="color: var(--primary);">📋 Execution Log:</strong><br><br>' + 
                        JSON.stringify(data.execution_log, null, 2);
                }}

            }} catch (error) {{
                reportContent.innerHTML = `
                    <div class="error-badge">
                        ❌ Error: ${{error.message}}
                    </div>
                    <p style="color: var(--text-dim); margin-top: 15px;">
                        ${{error.message.includes('timeout') ? 'The request timed out. Please try again.' : 'An unexpected error occurred.'}}
                    </p>
                `;
                console.error('Error:', error);
            }} finally {{
                btn.disabled = false;
                btnText.textContent = '🚀 Execute Mission';
                spinner.classList.remove('active');
            }}
        }}
        
        function formatMarkdown(text) {{
            return text
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*\\*(.+?)\\*\\*\\*/g, '<strong><em>$1</em></strong>')
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
                .replace(/^### (.+)$/gm, '<h3>$1</h3>')
                .replace(/^## (.+)$/gm, '<h2>$1</h2>')
                .replace(/^# (.+)$/gm, '<h2>$1</h2>')
                .replace(/^- (.+)$/gm, '<div style="margin-left: 20px;">• $1</div>')
                .replace(/^\\d+\\. (.+)$/gm, '<div style="margin-left: 20px;"># Secrets
cerebras_secret = modal.Secret.from_name("cerebras-api-</div>');
        }}
        
        function clearAll() {{
            document.getElementById('missionInput').value = '';
            document.getElementById('output').classList.remove('active');
        }}
        
        document.getElementById('missionInput').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') {{
                executeMission();
            }}
        }});
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@web_app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    try:
        from agent import IntelligenceSystem
        system = IntelligenceSystem()
        health = system.get_health_status()
        
        return HealthResponse(
            status=health["status"],
            uptime=health["uptime"],
            documents=health["documents"],
            cache_size=health["cache_size"],
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            uptime=0,
            documents=0,
            cache_size=0,
            timestamp=time.time()
        )

@web_app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        from agent import IntelligenceSystem
        system = IntelligenceSystem()
        health = system.get_health_status()
        
        return {
            "timestamp": time.time(),
            "system": health,
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MODAL FUNCTION FOR MISSION EXECUTION
# ============================================================================

@app.function(
    image=image,
    volumes={"/cache": volume},
    secrets=[cerebras_secret],
    container_idle_timeout=300,
    keep_warm=1,
    timeout=900,
    cpu=2.0,
    memory=4096
)
@web_app.post("/api/execute_mission")
def execute_mission(request: MissionRequest):
    """
    Execute intelligence mission
    
    This endpoint orchestrates autonomous mission execution by the AI agent.
    """
    try:
        logger.info(f"🎯 Mission received: {request.mission[:100]}...")
        
        from agent import IntelligenceSystem
        agent = IntelligenceSystem()
        
        # Execute mission (SYNC version - no await needed)
        result = agent.execute_mission({"mission": request.mission})
        
        logger.info(f"✅ Mission result: {result.get('success', False)}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

# ============================================================================
# ADMIN ENDPOINTS (Document Management)
# ============================================================================

@app.function(
    image=image,
    volumes={"/cache": volume},
    secrets=[cerebras_secret]
)
@web_app.post("/api/add_document")
def add_document(text: str, url: str = "", title: str = ""):
    """Add new document to database"""
    try:
        from agent import IntelligenceSystem
        agent = IntelligenceSystem()
        result = agent.add_document(text=text, url=url, title=title)
        return result
    except Exception as e:
        logger.error(f"Add document failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI entry point"""
    logger.info("🚀 Starting Intelligence Agent System v2.0...")
    return web_app
