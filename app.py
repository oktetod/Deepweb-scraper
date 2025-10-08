"""
FastAPI Application - Refactored Version
Production-ready with proper middleware, error handling, and monitoring
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import modal
import os
import sys
import time
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project directory to path
sys.path.append(os.path.dirname(__file__))

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

stub = modal.App(name="simple-web-agent")
volume = modal.Volume.persisted("intelligence-data-volume")
cerebras_secret = modal.Secret.from_name("cerebras-api-key")

# Docker image with pinned versions for reproducibility
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
    mission: str = Field(..., min_length=5, max_length=2000, description="Mission description")
    
    @validator('mission')
    def validate_mission(cls, v):
        if not v.strip():
            raise ValueError('Mission cannot be empty or whitespace only')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "mission": "Selidiki aktivitas terbaru grup ransomware 'LockBit'"
            }
        }


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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Application starting...")
    yield
    logger.info("👋 Application shutting down...")


web_app = FastAPI(
    title="Intelligence Agent System",
    description="Autonomous AI agent for intelligence missions",
    version="2.0.0",
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression middleware
web_app.add_middleware(GZipMiddleware, minimum_size=1000)


# Custom error handling middleware
@web_app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware"""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(e),
                "timestamp": time.time()
            }
        )


# Request logging middleware
@web_app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"← {request.method} {request.url.path} "
        f"[{response.status_code}] {duration:.3f}s"
    )
    
    return response


# Security headers middleware
@web_app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@web_app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": time.time()
        }
    )


@web_app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )


# ============================================================================
# FRONTEND UI
# ============================================================================

@web_app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def serve_frontend(request: Request):
    """
    Serve the main UI with modern, responsive design
    """
    api_url = str(request.url_for("execute_mission"))
    health_url = str(request.url_for("health_check"))
    
    html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligence Agent System v2.0</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        :root {{
            --primary: #2563eb;
            --primary-dark: #1e40af;
            --success: #10b981;
            --error: #ef4444;
            --bg: #0f172a;
            --surface: #1e293b;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --border: #334155;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #1e293b 100%);
            color: var(--text);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 20px;
            background: var(--surface);
            border-radius: 16px;
            border: 1px solid var(--border);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: var(--text-dim);
            font-size: 1.1rem;
        }}
        
        .card {{
            background: var(--surface);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        
        .status-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .status-indicator {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        textarea {{
            width: 100%;
            padding: 15px;
            background: var(--bg);
            border: 2px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
            resize: vertical;
            min-height: 120px;
            transition: all 0.3s;
        }}
        
        textarea:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }}
        
        .button-group {{
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }}
        
        button {{
            flex: 1;
            padding: 15px 30px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
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
        
        .btn-primary:hover {{
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
        }}
        
        .btn-primary:disabled {{
            background: var(--border);
            cursor: not-allowed;
            transform: none;
        }}
        
        .btn-secondary {{
            background: var(--surface);
            color: var(--text);
            border: 2px solid var(--border);
        }}
        
        .btn-secondary:hover {{
            border-color: var(--primary);
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
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        .output {{
            display: none;
            margin-top: 20px;
            padding: 20px;
            background: var(--bg);
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        
        .output.active {{ display: block; }}
        
        .output h3 {{
            color: var(--primary);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }}
        
        .report-content {{
            line-height: 1.8;
            color: var(--text-dim);
        }}
        
        .execution-log {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        .error-message {{
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--error);
            color: var(--error);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }}
        
        .success-message {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success);
            color: var(--success);
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{ font-size: 1.8rem; }}
            .button-group {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Intelligence Agent System</h1>
            <p>Autonomous AI-powered intelligence analysis platform</p>
        </div>
        
        <div class="status-bar" id="statusBar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>System Online</span>
            </div>
            <span id="systemInfo">Loading...</span>
        </div>
        
        <div class="card">
            <h2 style="margin-bottom: 15px;">Mission Control</h2>
            <p style="color: var(--text-dim); margin-bottom: 20px;">
                Enter your high-level mission objective. The agent will autonomously plan, 
                execute, and deliver a comprehensive intelligence report.
            </p>
            
            <textarea 
                id="missionInput" 
                placeholder="Example: Investigate recent activities of ransomware group 'LockBit' and analyze their attack patterns..."
                required
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
            <h3>📊 Mission Report</h3>
            <div class="report-content" id="reportContent"></div>
            <div class="execution-log" id="executionLog"></div>
        </div>
    </div>

    <script>
        const API_URL = "{api_url}";
        const HEALTH_URL = "{health_url}";
        
        // Check system health on load
        async function checkHealth() {{
            try {{
                const response = await fetch(HEALTH_URL);
                const data = await response.json();
                document.getElementById('systemInfo').textContent = 
                    `Documents: ${{data.documents}} | Uptime: ${{Math.floor(data.uptime)}}s`;
            }} catch (error) {{
                console.error('Health check failed:', error);
            }}
        }}
        
        checkHealth();
        setInterval(checkHealth, 30000); // Check every 30 seconds
        
        async function executeMission() {{
            const mission = document.getElementById('missionInput').value.trim();
            const outputDiv = document.getElementById('output');
            const reportContent = document.getElementById('reportContent');
            const executionLog = document.getElementById('executionLog');
            const executeBtn = document.getElementById('executeBtn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');

            if (!mission) {{
                alert('⚠️ Mission cannot be empty!');
                return;
            }}

            // Update UI
            executeBtn.disabled = true;
            btnText.textContent = 'Processing...';
            spinner.classList.add('active');
            outputDiv.classList.add('active');
            reportContent.innerHTML = '<p style="color: var(--text-dim);">🔄 Agent is analyzing and executing mission...</p>';
            executionLog.style.display = 'none';

            try {{
                const startTime = Date.now();
                
                const response = await fetch(API_URL, {{
                    method: 'POST',
                    headers: {{ 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }},
                    body: JSON.stringify({{ mission: mission }})
                }});

                const duration = ((Date.now() - startTime) / 1000).toFixed(2);

                if (!response.ok) {{
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP Error: ${{response.status}}`);
                }}

                const data = await response.json();
                
                // Display report
                if (data.success) {{
                    reportContent.innerHTML = `
                        <div class="success-message">
                            ✅ Mission completed successfully in ${{duration}}s
                        </div>
                        <div style="margin-top: 20px;">
                            ${{formatMarkdown(data.final_report)}}
                        </div>
                    `;
                }} else {{
                    reportContent.innerHTML = `
                        <div class="error-message">
                            ❌ Mission failed: ${{data.error || 'Unknown error'}}
                        </div>
                    `;
                }}
                
                // Display execution log
                if (data.execution_log && data.execution_log.length > 0) {{
                    executionLog.style.display = 'block';
                    executionLog.innerHTML = '<strong>Execution Log:</strong><br><br>' + 
                        JSON.stringify(data.execution_log, null, 2);
                }}

            }} catch (error) {{
                reportContent.innerHTML = `
                    <div class="error-message">
                        <strong>❌ Error:</strong> ${{error.message}}
                    </div>
                `;
                console.error('Error:', error);
            }} finally {{
                executeBtn.disabled = false;
                btnText.textContent = '🚀 Execute Mission';
                spinner.classList.remove('active');
            }}
        }}
        
        function formatMarkdown(text) {{
            // Simple markdown-to-HTML converter
            return text
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                .replace(/^### (.+)$/gm, '<h4>$1</h4>')
                .replace(/^## (.+)$/gm, '<h3>$1</h3>')
                .replace(/^# (.+)$/gm, '<h2>$1</h2>');
        }}
        
        function clearAll() {{
            document.getElementById('missionInput').value = '';
            document.getElementById('output').classList.remove('active');
        }}
        
        // Allow Ctrl+Enter to submit
        document.getElementById('missionInput').addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'Enter') {{
                executeMission();
            }}
        }});
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@web_app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint for monitoring
    """
    try:
        # Import here to avoid circular dependency
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System unhealthy"
        )


@web_app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get system metrics for monitoring
    """
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
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@modal.function(
    image=image,
    volumes={"/cache": volume},
    secrets=[cerebras_secret],
    container_idle_timeout=300,
    keep_warm=1,
    timeout=900,
    cpu=2.0,  # Allocate more CPU for better performance
)
@web_app.post("/api/execute_mission", tags=["Agent"])
async def execute_mission(request: MissionRequest):
    """
    Execute intelligence mission
    
    This endpoint receives a mission and orchestrates the autonomous execution
    by the intelligence system.
    """
    try:
        logger.info(f"Received mission: {request.mission[:100]}...")
        
        # Import agent system
        from agent import IntelligenceSystem
        agent_system = IntelligenceSystem()
        
        # Execute mission
        result = await agent_system.execute_mission({"mission": request.mission})
        
        logger.info(f"Mission completed: {result.get('success', False)}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Mission execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mission execution failed: {str(e)}"
        )


# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================

@modal.asgi_app()
def final_app():
    """
    Modal entry point for ASGI application
    """
    logger.info("🚀 Starting Intelligence Agent System...")
    return web_app
