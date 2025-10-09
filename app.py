"""
Intelligence Agent System - COMPLETE FIXED VERSION
All errors resolved, production ready
Version: 2.0.7-final
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from pathlib import Path
import modal
import shelve
import json
import time
import logging
import traceback
import os
import sys

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
CACHE_DIR = "/cache"
DB_DIR = Path(f"{CACHE_DIR}/db")
DOC_STORE_PATH = str(DB_DIR / "documents.db")
FAISS_INDEX_PATH = str(DB_DIR / "vectors.faiss")
VECTOR_DIMENSION = 384
MODEL_NAME = "all-MiniLM-L6-v2"

db_lock = Lock()
index_lock = Lock()
SYSTEM_START_TIME = time.time()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def retry_with_exponential_backoff(max_retries=3, initial_delay=1.0, exponential_base=2.0, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Final retry failed for {func.__name__}: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= exponential_base
        return wrapper
    return decorator

@contextmanager
def safe_shelve_open(path: str, flag: str = 'r'):
    db = None
    try:
        with db_lock:
            db = shelve.open(path, flag=flag)
            yield db
    except Exception as e:
        logger.error(f"Shelve operation error: {e}")
        raise
    finally:
        if db is not None:
            try:
                db.close()
            except Exception as e:
                logger.warning(f"Error closing shelve: {e}")

class ValidationError(Exception):
    pass

# ============================================================================
# CEREBRAS CLIENT - FIXED VERSION
# ============================================================================

class CerebrasClient:
    """
    Fixed Cerebras client with multiple fallback strategies
    """
    def __init__(self):
        self.api_keys = [
            os.environ.get("CEREBRAS_API_KEY", ""),
            os.environ.get("CEREBRAS_API_KEY_2", "")
        ]
        self.api_keys = [key for key in self.api_keys if key]
        
        if not self.api_keys:
            raise ValueError("❌ No valid Cerebras API keys found")
        
        # API configuration
        self.base_url = "https://api.cerebras.ai/v1/chat/completions"
        self.use_sdk = False
        
        # Try SDK first, fallback to requests
        try:
            from cerebras.cloud.sdk import Cerebras
            self.Cerebras = Cerebras
            self.use_sdk = True
            logger.info("✅ Using Cerebras SDK")
        except ImportError as e:
            logger.warning(f"⚠️ Cerebras SDK not available, using direct API: {e}")
            import requests
            self.requests = requests
        
        logger.info(f"🔑 Loaded {len(self.api_keys)} API key(s)")
        logger.info(f"🌐 API URL: {self.base_url}")
    
    @retry_with_exponential_backoff(max_retries=2, initial_delay=0.5)
    def call(self, **kwargs) -> Any:
        """Main call method with fallback"""
        if self.use_sdk:
            try:
                return self._sdk_call(**kwargs)
            except Exception as e:
                logger.error(f"❌ SDK call failed: {e}")
                logger.info("🔄 Switching to direct API mode")
                self.use_sdk = False
                return self._direct_call(**kwargs)
        else:
            return self._direct_call(**kwargs)
    
    def _sdk_call(self, **kwargs) -> Any:
        """Call using Cerebras SDK"""
        last_error = None
        
        for i, api_key in enumerate(self.api_keys):
            try:
                logger.info(f"🔑 SDK call attempt #{i+1}")
                client = self.Cerebras(api_key=api_key)
                response = client.chat.completions.create(**kwargs)
                logger.info(f"✅ SDK call successful")
                return response
            except Exception as e:
                error_msg = str(e)[:200]
                logger.error(f"❌ SDK key #{i+1} failed: {error_msg}")
                last_error = e
                continue
        
        raise last_error or Exception("All SDK attempts failed")
    
    def _direct_call(self, **kwargs) -> Any:
        """Direct API call using requests - FIXED VERSION"""
        import requests
        
        # Extract parameters
        model = kwargs.get("model", "llama-4-maverick-17b-128e-instruct")
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature", 0.3)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        # Valid models
        valid_models = [
            "llama-4-maverick-17b-128e-instruct",
            "llama3.1-8b",
            "llama3.1-70b"
        ]
        
        if model not in valid_models:
            logger.warning(f"⚠️ Model '{model}' may not be valid. Using default.")
            model = "llama-4-maverick-17b-128e-instruct"
        
        last_error = None
        
        for i, api_key in enumerate(self.api_keys):
            try:
                logger.info(f"🔑 Direct API attempt #{i+1}")
                logger.info(f"📡 Model: {model}")
                logger.info(f"💬 Messages: {len(messages)}")
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                logger.info(f"📥 Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"❌ Response body: {response.text[:500]}")
                
                response.raise_for_status()
                data = response.json()
                
                if 'choices' not in data or not data['choices']:
                    raise ValueError("Invalid API response: missing choices")
                
                # Create SDK-compatible response
                class DirectResponse:
                    def __init__(self, data):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': data['choices'][0]['message']['content']
                            })()
                        })()]
                
                logger.info(f"✅ Direct API call successful")
                return DirectResponse(data)
                
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                error_body = e.response.text[:500]
                
                if status == 401:
                    logger.warning(f"🔑 Key #{i+1} unauthorized")
                elif status == 404:
                    logger.error(f"❌ 404 Not Found - Invalid endpoint or model")
                    logger.error(f"📍 URL: {self.base_url}")
                    logger.error(f"📦 Model: {model}")
                elif status == 429:
                    logger.warning(f"⏱️ Rate limit hit on key #{i+1}")
                else:
                    logger.error(f"❌ HTTP {status}: {error_body}")
                
                last_error = e
                continue
                
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                last_error = e
                continue
        
        if last_error:
            raise last_error
        else:
            raise Exception("All API keys exhausted")

# ============================================================================
# EMBEDDING MANAGER
# ============================================================================

class EmbeddingManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("🔧 Initializing embedding model...")
            self.model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
            self.dimension = VECTOR_DIMENSION
            self._initialized = True
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    @retry_with_exponential_backoff(max_retries=3)
    def encode(self, texts: List[str]) -> Any:
        import numpy as np
        return self.model.encode(texts, convert_to_numpy=True)

# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    def __init__(self):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed")
        
        logger.info("🗄️ Initializing FAISS vector store...")
        self.dimension = VECTOR_DIMENSION
        
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                with index_lock:
                    self.index = faiss.read_index(FAISS_INDEX_PATH)
                logger.info(f"✅ Loaded index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
        except Exception as e:
            logger.error(f"⚠️ Failed to load index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        import faiss
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with index_lock:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        logger.info("✅ Created new FAISS index")
    
    def add_vectors(self, vectors: Any, ids: Any):
        import numpy as np
        with index_lock:
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors_np, ids_np)
    
    def search(self, query_vector: Any, k: int = 3) -> Tuple[Any, Any]:
        import numpy as np
        with index_lock:
            if self.index.ntotal == 0:
                return np.array([[]], dtype=np.float32), np.array([[-1]], dtype=np.int64)
            return self.index.search(np.array([query_vector], dtype=np.float32), min(k, self.index.ntotal))
    
    def get_total_vectors(self) -> int:
        with index_lock:
            return self.index.ntotal
    
    def save(self):
        import faiss
        try:
            with index_lock:
                faiss.write_index(self.index, FAISS_INDEX_PATH)
            logger.info("✅ FAISS index saved")
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")

# ============================================================================
# INPUT VALIDATOR
# ============================================================================

class InputValidator:
    @staticmethod
    def validate_mission(mission: str) -> str:
        if not mission or not isinstance(mission, str):
            raise ValidationError("Mission must be a non-empty string")
        mission = mission.strip()
        if len(mission) < 10:
            raise ValidationError("Mission too short (minimum 10 characters)")
        if len(mission) > 2000:
            raise ValidationError("Mission too long (maximum 2000 characters)")
        return mission
    
    @staticmethod
    def validate_doc_id(doc_id: Any) -> int:
        try:
            doc_id = int(doc_id)
            if doc_id < 0:
                raise ValidationError("Document ID must be non-negative")
            return doc_id
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid document ID: {doc_id}")
    
    @staticmethod
    def validate_query(query: str) -> str:
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")
        query = query.strip()
        if len(query) < 3:
            raise ValidationError("Query too short (minimum 3 characters)")
        return query

# ============================================================================
# INTELLIGENCE SYSTEM
# ============================================================================

class IntelligenceSystem:
    def __init__(self):
        logger.info("🚀 Initializing Intelligence System...")
        
        try:
            self.embedding_manager = EmbeddingManager()
            self.vector_store = VectorStoreManager()
            self.cerebras_client = CerebrasClient()
            self.validator = InputValidator()
            logger.info(f"✅ System ready with {self.vector_store.get_total_vectors()} documents")
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}\n{traceback.format_exc()}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        try:
            total_docs = self.vector_store.get_total_vectors()
            uptime = time.time() - SYSTEM_START_TIME
            cache_size = 0
            if DB_DIR.exists():
                for file in DB_DIR.iterdir():
                    cache_size += file.stat().st_size
            
            cerebras_mode = "SDK" if self.cerebras_client.use_sdk else "Direct API"
            
            return {
                "status": "healthy",
                "uptime": uptime,
                "documents": total_docs,
                "cache_size": cache_size,
                "vector_dimension": VECTOR_DIMENSION,
                "model": MODEL_NAME,
                "cerebras_mode": cerebras_mode,
                "api_url": self.cerebras_client.base_url,
                "python_version": sys.version
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "uptime": 0, "documents": 0}
    
    def semantic_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            query = self.validator.validate_query(query)
            k = min(max(k, 1), 10)
            
            if self.vector_store.get_total_vectors() == 0:
                return {"status": "error", "detail": "Database is empty", "results": []}
            
            query_vector = self.embedding_manager.encode([query])[0]
            distances, ids = self.vector_store.search(query_vector, k)
            
            results = []
            with safe_shelve_open(DOC_STORE_PATH) as doc_store:
                for i, doc_id in enumerate(ids[0]):
                    if doc_id != -1:
                        doc_key = str(doc_id)
                        if doc_key in doc_store:
                            doc_info = doc_store[doc_key]
                            results.append({
                                "doc_id": int(doc_id),
                                "url": doc_info.get("url", "N/A"),
                                "title": doc_info.get("title", "Untitled"),
                                "score": float(1 - distances[0][i]),
                                "snippet": doc_info.get("text", "")[:200] + "..."
                            })
            
            logger.info(f"✅ Search completed: {len(results)} results")
            return {"status": "success", "results": results, "total_found": len(results)}
        except ValidationError as e:
            return {"status": "error", "detail": str(e), "results": []}
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {"status": "error", "detail": str(e), "results": []}
    
    def get_document_content(self, doc_id: int) -> Dict[str, Any]:
        try:
            doc_id = self.validator.validate_doc_id(doc_id)
            with safe_shelve_open(DOC_STORE_PATH) as doc_store:
                doc_key = str(doc_id)
                if doc_key not in doc_store:
                    return {"status": "error", "detail": f"Document {doc_id} not found"}
                doc_data = doc_store[doc_key]
                return {
                    "status": "success",
                    "doc_id": doc_id,
                    "content": doc_data.get("text", ""),
                    "url": doc_data.get("url", "N/A"),
                    "title": doc_data.get("title", "Untitled"),
                    "length": len(doc_data.get("text", ""))
                }
        except ValidationError as e:
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    
    def summarize_with_maverick(self, text: str) -> str:
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text must be a non-empty string")
            max_length = 8000
            if len(text) > max_length:
                text = text[:max_length] + "..."
            prompt = (
                "You are an efficient AI assistant. Create a concise one-paragraph summary "
                "of the following text. Output ONLY the summary.\n\n"
                f"Text: {text}"
            )
            response = self.cerebras_client.call(
                model="llama-4-maverick-17b-128e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            return f"[Summarization failed: {str(e)}]"
    
    def add_document(self, text: str, url: str = "", title: str = "") -> Dict[str, Any]:
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text must be a non-empty string")
            text = text.strip()
            if len(text) < 50:
                raise ValidationError("Text too short (minimum 50 characters)")
            
            vector = self.embedding_manager.encode([text])[0]
            doc_id = self.vector_store.get_total_vectors()
            self.vector_store.add_vectors([vector], [doc_id])
            
            with safe_shelve_open(DOC_STORE_PATH, flag='c') as doc_store:
                doc_store[str(doc_id)] = {
                    "text": text,
                    "url": url or "N/A",
                    "title": title or f"Document {doc_id}",
                    "added_at": time.time()
                }
            
            self.vector_store.save()
            logger.info(f"✅ Document {doc_id} added")
            return {"status": "success", "doc_id": doc_id, "message": f"Document added with ID {doc_id}"}
        except ValidationError as e:
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    
    def execute_mission(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        try:
            mission = mission_data.get("mission")
            if not mission:
                raise ValidationError("Mission field is required")
            mission = self.validator.validate_mission(mission)
            logger.info(f"📋 Mission: '{mission[:100]}...'")
            
            # Step 1: Planning
            architect_prompt = """You are "Architect", an AI planner.
Convert missions into JSON action plans.

Available tools:
1. semantic_search(query: str, k: int=3) - Search documents
2. get_document_content(doc_id: int) - Get full document
3. summarize_with_maverick(text: str) - Summarize text

Output ONLY a JSON array.
Use placeholders like "RESULT_FROM_STEP_1.results[0].doc_id" for dependencies.

Example:
[
  {
    "tool_name": "semantic_search",
    "parameters": {"query": "ransomware", "k": 3},
    "description": "Search for ransomware documents"
  }
]"""
            
            response = self.cerebras_client.call(
                messages=[
                    {"role": "system", "content": architect_prompt},
                    {"role": "user", "content": f"Mission: {mission}"}
                ],
                model="llama-4-maverick-17b-128e-instruct",
                temperature=0.1,
                max_tokens=2000
            )
            
            plan_str = response.choices[0].message.content
            plan = self._parse_plan(plan_str)
            logger.info(f"🗺️ Plan: {len(plan)} steps")
            
            # Step 2: Execute
            execution_log = []
            context = {}
            available_tools = {
                "semantic_search": self.semantic_search,
                "get_document_content": self.get_document_content,
                "summarize_with_maverick": self.summarize_with_maverick
            }
            
            for i, step in enumerate(plan):
                step_start = time.time()
                tool_name = step.get("tool_name")
                parameters = step.get("parameters", {})
                
                resolved_params = {}
                for key, value in parameters.items():
                    try:
                        resolved_params[key] = self._resolve_placeholder(value, context)
                    except Exception as e:
                        resolved_params[key] = None
                
                if tool_name not in available_tools:
                    result = {"status": "error", "detail": f"Tool '{tool_name}' not found"}
                else:
                    try:
                        result = available_tools[tool_name](**resolved_params)
                    except Exception as e:
                        result = {"status": "error", "detail": str(e)}
                
                context[f"step_{i+1}"] = result
                execution_log.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "description": step.get("description", ""),
                    "result": result,
                    "duration_seconds": round(time.time() - step_start, 2)
                })
            
            # Step 3: Generate Report
            report_prompt = f"""You are an intelligence analyst. Write a report for: '{mission}'.

Execution Log:
{json.dumps(execution_log, indent=2)}

Write in Markdown with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions"""
            
            response = self.cerebras_client.call(
                messages=[{"role": "user", "content": report_prompt}],
                model="llama-4-maverick-17b-128e-instruct",
                temperature=0.5,
                max_tokens=3000
            )
            
            final_report = response.choices[0].message.content
            total_duration = time.time() - start_time
            
            return {
                "success": True,
                "status": "success",
                "mission": mission,
                "final_report": final_report,
                "execution_log": execution_log,
                "metadata": {
                    "total_steps": len(plan),
                    "successful_steps": sum(1 for log in execution_log if log.get("result", {}).get("status") == "success"),
                    "total_duration_seconds": round(total_duration, 2)
                }
            }
        except Exception as e:
            logger.error(f"❌ Mission failed: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "mission": mission_data.get("mission", ""),
                "final_report": f"## Mission Failed\n\n**Error:** {str(e)}",
                "execution_log": []
            }
    
    def _parse_plan(self, plan_str: str) -> List[Dict]:
        try:
            if plan_str.strip().startswith("```"):
                lines = plan_str.strip().split('\n')
                plan_str = '\n'.join(lines[1:-1]) if len(lines) > 2 else plan_str
            return json.loads(plan_str)
        except Exception as e:
            raise ValidationError(f"Invalid JSON: {str(e)}")
    
    def _resolve_placeholder(self, value: Any, context: Dict) -> Any:
        if not isinstance(value, str) or not value.startswith("RESULT_FROM_STEP_"):
            return value
        import re
        match = re.search(r'RESULT_FROM_STEP_(\d+)', value)
        if not match:
            raise ValueError(f"Invalid placeholder: {value}")
        step_num = int(match.group(1))
        result_key = f"step_{step_num}"
        if result_key not in context:
            raise ValueError(f"Step {step_num} not found")
        result = context[result_key]
        path = value.replace(f"RESULT_FROM_STEP_{step_num}", "").strip().lstrip('.').lstrip('[')
        if path:
            parts = re.split(r'[\.\[\]]', path)
            parts = [p for p in parts if p]
            for part in parts:
                if part.isdigit():
                    result = result[int(part)]
                elif part:
                    result = result.get(part) if isinstance(result, dict) else getattr(result, part, None)
                if result is None:
                    raise ValueError(f"Path '{part}' not found")
        return result

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

app = modal.App(name="intelligence-agent-system-fixed")
volume = modal.Volume.from_name("intelligence-data-vol", create_if_missing=True)
cerebras_secret = modal.Secret.from_name("cerebras-api-key")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "numpy",
        "torch",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "requests"
    )
    .run_commands("pip install cerebras-cloud-sdk --no-deps --force-reinstall || echo 'SDK install failed, will use API'")
)

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

web_app = FastAPI(title="Intelligence Agent System", version="2.0.7-final")
web_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
web_app.add_middleware(GZipMiddleware, minimum_size=1000)

class MissionRequest(BaseModel):
    mission: str = Field(..., min_length=10, max_length=2000)

class DocumentRequest(BaseModel):
    text: str = Field(..., min_length=50)
    url: str = ""
    title: str = ""

@web_app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    api_url = str(request.url_for("execute_mission"))
    health_url = str(request.url_for("health_check"))
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligence Agent System</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --primary: #3b82f6;
            --success: #10b981;
            --error: #ef4444;
            --bg: #0f172a;
            --surface: #1e293b;
            --text: #f1f5f9;
            --border: #334155;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, var(--bg), #1e293b);
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
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, var(--primary), var(--success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .card {{
            background: var(--surface);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 25px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        .card h2 {{ color: var(--primary); margin-bottom: 20px; }}
        textarea {{
            width: 100%;
            padding: 16px;
            background: var(--bg);
            border: 2px solid var(--border);
            border-radius: 10px;
            color: var(--text);
            font-size: 1rem;
            min-height: 140px;
            resize: vertical;
        }}
        textarea:focus {{ outline: none; border-color: var(--primary); }}
        button {{
            width: 100%;
            padding: 16px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 20px;
            background: var(--primary);
            color: white;
            transition: all 0.3s;
        }}
        button:hover:not(:disabled) {{
            background: #2563eb;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }}
        button:disabled {{ background: #475569; cursor: not-allowed; }}
        .spinner {{
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-left: 10px;
        }}
        .spinner.active {{ display: inline-block; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .output {{ display: none; animation: fadeIn 0.5s; }}
        .output.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        .status-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-top: 10px;
        }}
        .status-success {{ background: #10b98144; color: #10b981; }}
        .status-error {{ background: #ef444444; color: #ef4444; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Intelligence Agent System</h1>
            <p style="color: #94a3b8; margin-top: 10px;">Autonomous AI-powered intelligence analysis</p>
            <p style="color: #64748b; font-size: 0.875rem;">v2.0.7-final | Production Ready</p>
            <div id="systemStatus"></div>
        </div>
        
        <div class="card">
            <h2>🎯 Mission Control</h2>
            <textarea id="missionInput" placeholder="Example: Investigate LockBit ransomware activities and recent attack campaigns..."></textarea>
            <button onclick="executeMission()" id="executeBtn">
                <span id="btnText">🚀 Execute Mission</span>
                <div class="spinner" id="spinner"></div>
            </button>
        </div>
        
        <div class="output" id="output">
            <div class="card">
                <h2>📊 Mission Report</h2>
                <div id="reportContent" style="line-height: 1.6;"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Check system health
        fetch("{health_url}")
            .then(r => r.json())
            .then(data => {{
                let badge = '';
                if (data.status === 'healthy') {{
                    const mode = data.cerebras_mode || 'Unknown';
                    badge = `<span class="status-badge status-success">✅ System Online (${{mode}})</span>`;
                    badge += `<br><small style="color:#64748b;font-size:0.8rem;">Documents: ${{data.documents}} | Uptime: ${{Math.round(data.uptime)}}s</small>`;
                }} else {{
                    badge = '<span class="status-badge status-error">⚠️ System Offline</span>';
                }}
                document.getElementById('systemStatus').innerHTML = badge;
            }})
            .catch(e => {{
                document.getElementById('systemStatus').innerHTML = 
                    '<span class="status-badge status-error">⚠️ Cannot connect</span>';
            }});
        
        async function executeMission() {{
            const mission = document.getElementById('missionInput').value.trim();
            
            if (!mission || mission.length < 10) {{
                alert('❌ Mission must be at least 10 characters');
                return;
            }}
            
            const btn = document.getElementById('executeBtn');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btnText');
            
            btn.disabled = true;
            spinner.classList.add('active');
            btnText.textContent = '🔄 Processing...';
            
            document.getElementById('output').classList.add('active');
            document.getElementById('reportContent').innerHTML = 
                '<p style="text-align:center;color:#64748b;">🔄 AI agent is planning and executing mission...<br><small>This may take 30-60 seconds</small></p>';
            
            try {{
                const res = await fetch("{api_url}", {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{mission}})
                }});
                
                const data = await res.json();
                
                if (data.success) {{
                    const report = data.final_report || '';
                    const meta = data.metadata || {{}};
                    const log = data.execution_log || [];
                    
                    let logHtml = '<details style="margin-top:20px;"><summary style="cursor:pointer;color:var(--primary);font-weight:600;">📋 Execution Log</summary><div style="margin-top:10px;">';
                    log.forEach(step => {{
                        const icon = step.result?.status === 'success' ? '✅' : '❌';
                        logHtml += `<div style="margin:10px 0;padding:10px;background:var(--bg);border-radius:6px;">`;
                        logHtml += `<strong>${{icon}} Step ${{step.step}}: ${{step.tool}}</strong><br>`;
                        logHtml += `<small style="color:#94a3b8;">${{step.description}}</small><br>`;
                        logHtml += `<small style="color:#64748b;">Duration: ${{step.duration_seconds}}s</small>`;
                        logHtml += `</div>`;
                    }});
                    logHtml += '</div></details>';
                    
                    let footer = '<hr style="margin:20px 0;border-color:var(--border);">';
                    footer += `<p style="color:#64748b;font-size:0.875rem;">`;
                    footer += `⏱️ Completed in ${{meta.total_duration_seconds || 0}}s | `;
                    footer += `✅ ${{meta.successful_steps || 0}}/${{meta.total_steps || 0}} steps successful`;
                    footer += `</p>`;
                    
                    const formatted = report
                        .replace(/\\n/g, '<br>')
                        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                        .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                        .replace(/^### (.+)$/gm, '<h3 style="color:var(--primary);margin:15px 0 10px;">$1</h3>')
                        .replace(/^## (.+)$/gm, '<h2 style="color:var(--success);margin:20px 0 15px;">$1</h2>');
                    
                    document.getElementById('reportContent').innerHTML = 
                        '<div style="color:var(--success);font-weight:600;margin-bottom:20px;">✅ Mission Complete</div>' +
                        formatted + footer + logHtml;
                }} else {{
                    const error = data.error || 'Unknown error';
                    document.getElementById('reportContent').innerHTML = 
                        `<div style="color:var(--error);">❌ <strong>Mission Failed</strong><br><br>Error: ${{error}}<br><br><small>Please check system logs or try again.</small></div>`;
                }}
            }} catch (e) {{
                document.getElementById('reportContent').innerHTML = 
                    `<div style="color:var(--error);">❌ <strong>Network Error</strong><br><br>${{e.message}}</div>`;
            }} finally {{
                btn.disabled = false;
                spinner.classList.remove('active');
                btnText.textContent = '🚀 Execute Mission';
            }}
        }}
        
        // Keyboard shortcut: Ctrl+Enter to submit
        document.getElementById('missionInput').addEventListener('keydown', (e) => {{
            if (e.ctrlKey && e.key === 'Enter') {{
                executeMission();
            }}
        }});
    </script>
</body>
</html>"""
    
    return HTMLResponse(content=html_content)

@web_app.post("/api/execute_mission")
async def execute_mission(request: MissionRequest):
    """Execute intelligence mission"""
    try:
        system = IntelligenceSystem()
        result = system.execute_mission({"mission": request.mission})
        return result
    except Exception as e:
        logger.error(f"Mission execution failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        system = IntelligenceSystem()
        return system.get_health_status()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@web_app.post("/api/add_document")
async def add_document(request: DocumentRequest):
    """Add document to knowledge base"""
    try:
        system = IntelligenceSystem()
        result = system.add_document(request.text, request.url, request.title)
        return result
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        system = IntelligenceSystem()
        health = system.get_health_status()
        return {
            "timestamp": time.time(),
            "system": health,
            "version": "2.0.7-final"
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================

@app.function(
    image=image,
    secrets=[cerebras_secret],
    volumes={CACHE_DIR: volume},
    cpu=2.0,
    memory=4096,
    timeout=900,
    keep_warm=1,
    container_idle_timeout=300
)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI app"""
    return web_app

# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="0.0.0.0", port=8000)
