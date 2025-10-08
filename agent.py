# agent.py (Production-Ready - Fixed & Enhanced)

import os
import shelve
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager
from functools import lru_cache, wraps
from threading import Lock
import traceback

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CACHE_DIR = "/cache"
DB_DIR = Path(f"{CACHE_DIR}/db")
DOC_STORE_PATH = str(DB_DIR / "documents.db")
FAISS_INDEX_PATH = str(DB_DIR / "vectors.faiss")
VECTOR_DIMENSION = 384
MODEL_NAME = "all-MiniLM-L6-v2"

# Thread-safe locks
db_lock = Lock()
index_lock = Lock()

# System start time for uptime tracking
SYSTEM_START_TIME = time.time()


# --- Utility Functions ---
def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """Decorator untuk retry dengan exponential backoff"""
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
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= exponential_base
            return None
        return wrapper
    return decorator


@contextmanager
def safe_shelve_open(path: str, flag: str = 'r'):
    """Context manager untuk thread-safe shelve operations"""
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
    """Custom exception untuk validation errors"""
    pass


# --- Core Components ---
class EmbeddingManager:
    """Mengelola model embedding dengan caching dan lazy loading"""
    
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
            
            logger.info("🔧 Initializing embedding model (singleton)...")
            self.model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
            self.dimension = VECTOR_DIMENSION
            self._initialized = True
            logger.info("✅ Embedding model loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import sentence_transformers: {e}")
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    @retry_with_exponential_backoff(max_retries=3)
    def encode(self, texts: List[str]) -> Any:
        """Encode texts dengan retry mechanism"""
        try:
            import numpy as np
            return self.model.encode(texts, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise


class VectorStoreManager:
    """Mengelola FAISS index dengan thread-safety"""
    
    def __init__(self):
        try:
            import faiss
        except ImportError:
            logger.error("❌ FAISS not installed")
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")
        
        logger.info("🗄️ Initializing FAISS vector store...")
        self.dimension = VECTOR_DIMENSION
        
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                with index_lock:
                    self.index = faiss.read_index(FAISS_INDEX_PATH)
                logger.info(f"✅ Loaded existing index with {self.index.ntotal} vectors")
            else:
                self._create_new_index()
        except Exception as e:
            logger.error(f"⚠️ Failed to load index, creating new one. Error: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        import faiss
        
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with index_lock:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
        logger.info("✅ Created new FAISS index")
    
    def add_vectors(self, vectors: Any, ids: Any):
        """Add vectors to index"""
        import numpy as np
        
        with index_lock:
            vectors_np = np.array(vectors, dtype=np.float32)
            ids_np = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors_np, ids_np)
    
    def search(self, query_vector: Any, k: int = 3) -> Tuple[Any, Any]:
        """Thread-safe search"""
        import numpy as np
        
        with index_lock:
            if self.index.ntotal == 0:
                return np.array([[]], dtype=np.float32), np.array([[-1]], dtype=np.int64)
            return self.index.search(np.array([query_vector], dtype=np.float32), min(k, self.index.ntotal))
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors"""
        with index_lock:
            return self.index.ntotal
    
    def save(self):
        """Save index to disk"""
        import faiss
        
        try:
            with index_lock:
                faiss.write_index(self.index, FAISS_INDEX_PATH)
            logger.info("✅ FAISS index saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index: {e}")


class CerebrasClient:
    """Wrapper untuk Cerebras API dengan fallback dan retry logic"""
    
    def __init__(self):
        self.api_keys = [
            os.environ.get("CEREBRAS_API_KEY"),
            os.environ.get("CEREBRAS_API_KEY_2")
        ]
        self.api_keys = [key for key in self.api_keys if key]
        
        if not self.api_keys:
            raise ValueError("❌ No valid Cerebras API keys found in environment")
        
        logger.info(f"🔑 Initialized with {len(self.api_keys)} API key(s)")
    
    @retry_with_exponential_backoff(max_retries=2, initial_delay=0.5)
    def call(self, **kwargs) -> Any:
        """Call Cerebras API dengan automatic fallback"""
        try:
            from cerebras.cloud.sdk import Cerebras
            from cerebras.cloud.sdk.errors import AuthenticationError, APIError
        except ImportError:
            logger.error("❌ Cerebras SDK not installed")
            raise ImportError("cerebras-cloud-sdk not installed. Run: pip install cerebras-cloud-sdk")
        
        last_error = None
        
        for i, api_key in enumerate(self.api_keys):
            try:
                logger.info(f"🔄 Attempting API call with key #{i+1}")
                client = Cerebras(api_key=api_key)
                response = client.chat.completions.create(**kwargs)
                logger.info(f"✅ API call successful with key #{i+1}")
                return response
                
            except AuthenticationError as e:
                logger.warning(f"⚠️ Authentication failed with key #{i+1}: {str(e)[:100]}")
                last_error = e
                continue
                
            except APIError as e:
                logger.error(f"❌ API Error with key #{i+1}: {str(e)[:100]}")
                last_error = e
                raise
                
            except Exception as e:
                logger.error(f"❌ Unexpected error with key #{i+1}: {str(e)[:100]}")
                last_error = e
                raise
        
        raise Exception(f"All {len(self.api_keys)} API key(s) failed. Last error: {last_error}")


class InputValidator:
    """Validasi dan sanitasi input dari user"""
    
    @staticmethod
    def validate_mission(mission: str) -> str:
        """Validate mission input"""
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
        """Validate document ID"""
        try:
            doc_id = int(doc_id)
            if doc_id < 0:
                raise ValidationError("Document ID must be non-negative")
            return doc_id
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid document ID: {doc_id}")
    
    @staticmethod
    def validate_query(query: str) -> str:
        """Validate search query"""
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")
        
        query = query.strip()
        
        if len(query) < 3:
            raise ValidationError("Query too short (minimum 3 characters)")
        
        return query


# --- Main Intelligence System ---
class IntelligenceSystem:
    """
    Production-ready intelligence system dengan comprehensive error handling,
    resource management, dan monitoring capabilities.
    """
    
    def __init__(self):
        """Initialize system dengan lazy loading dan proper error handling"""
        logger.info("🚀 Initializing Intelligence System...")
        
        try:
            # Singleton components
            self.embedding_manager = EmbeddingManager()
            self.vector_store = VectorStoreManager()
            self.cerebras_client = CerebrasClient()
            self.validator = InputValidator()
            
            logger.info(f"✅ System ready with {self.vector_store.get_total_vectors()} documents")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}\n{traceback.format_exc()}")
            raise
    
    # --- Health & Monitoring ---
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            total_docs = self.vector_store.get_total_vectors()
            uptime = time.time() - SYSTEM_START_TIME
            
            # Get cache size
            cache_size = 0
            if DB_DIR.exists():
                for file in DB_DIR.iterdir():
                    cache_size += file.stat().st_size
            
            return {
                "status": "healthy",
                "uptime": uptime,
                "documents": total_docs,
                "cache_size": cache_size,
                "vector_dimension": VECTOR_DIMENSION,
                "model": MODEL_NAME
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "uptime": time.time() - SYSTEM_START_TIME,
                "documents": 0,
                "cache_size": 0
            }
    
    # --- Tools ---
    
    def semantic_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Semantic search dengan comprehensive error handling dan validation.
        """
        try:
            query = self.validator.validate_query(query)
            
            if k < 1 or k > 10:
                k = min(max(k, 1), 10)
                logger.warning(f"k adjusted to valid range: {k}")
            
            if self.vector_store.get_total_vectors() == 0:
                return {
                    "status": "error",
                    "detail": "Database is empty. No documents to search.",
                    "results": []
                }
            
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
            
            return {
                "status": "success",
                "results": results,
                "total_found": len(results)
            }
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return {"status": "error", "detail": str(e), "results": []}
        except Exception as e:
            logger.error(f"❌ Search failed: {e}\n{traceback.format_exc()}")
            return {"status": "error", "detail": f"Search failed: {str(e)}", "results": []}
    
    def get_document_content(self, doc_id: int) -> Dict[str, Any]:
        """Retrieve full document content"""
        try:
            doc_id = self.validator.validate_doc_id(doc_id)
            
            with safe_shelve_open(DOC_STORE_PATH) as doc_store:
                doc_key = str(doc_id)
                
                if doc_key not in doc_store:
                    logger.warning(f"Document {doc_id} not found")
                    return {
                        "status": "error",
                        "detail": f"Document with ID {doc_id} not found"
                    }
                
                doc_data = doc_store[doc_key]
                logger.info(f"✅ Retrieved document {doc_id}")
                
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
            logger.error(f"❌ Failed to retrieve document: {e}")
            return {"status": "error", "detail": str(e)}
    
    def summarize_with_maverick(self, text: str) -> str:
        """Summarize text using Llama Maverick"""
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text must be a non-empty string")
            
            max_length = 8000
            if len(text) > max_length:
                text = text[:max_length] + "..."
                logger.warning(f"Text truncated to {max_length} characters")
            
            prompt = (
                "You are an efficient AI assistant. Create a concise one-paragraph summary "
                "of the following text. Output ONLY the summary without any preamble.\n\n"
                f"Text: {text}"
            )
            
            response = self.cerebras_client.call(
                model="llama-4-maverick-17b-128e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"✅ Summarization completed")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            return f"[Summarization failed: {str(e)}]"
    
    def add_document(self, text: str, url: str = "", title: str = "") -> Dict[str, Any]:
        """Add new document to database"""
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text must be a non-empty string")
            
            text = text.strip()
            if len(text) < 50:
                raise ValidationError("Text too short (minimum 50 characters)")
            
            # Generate embedding
            vector = self.embedding_manager.encode([text])[0]
            
            # Get next ID
            doc_id = self.vector_store.get_total_vectors()
            
            # Add to FAISS
            self.vector_store.add_vectors([vector], [doc_id])
            
            # Save metadata
            with safe_shelve_open(DOC_STORE_PATH, flag='c') as doc_store:
                doc_store[str(doc_id)] = {
                    "text": text,
                    "url": url or "N/A",
                    "title": title or f"Document {doc_id}",
                    "added_at": time.time()
                }
            
            # Save index
            self.vector_store.save()
            
            logger.info(f"✅ Document {doc_id} added successfully")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": f"Document added with ID {doc_id}"
            }
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            return {"status": "error", "detail": str(e)}
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return {"status": "error", "detail": str(e)}
    
    # --- Mission Execution ---
    
    def _parse_plan_from_architect(self, plan_str: str) -> List[Dict[str, Any]]:
        """Parse and validate plan"""
        try:
            if plan_str.strip().startswith("```"):
                lines = plan_str.strip().split('\n')
                plan_str = '\n'.join(lines[1:-1]) if len(lines) > 2 else plan_str
            
            plan = json.loads(plan_str)
            
            if not isinstance(plan, list):
                raise ValidationError("Plan must be a JSON array")
            
            for i, step in enumerate(plan):
                if not isinstance(step, dict):
                    raise ValidationError(f"Step {i+1} must be a JSON object")
                if "tool_name" not in step:
                    raise ValidationError(f"Step {i+1} missing 'tool_name'")
            
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise ValidationError(f"Invalid JSON: {str(e)}")
    
    def _resolve_placeholder(self, value: Any, context: Dict[str, Any]) -> Any:
        """Resolve placeholder dari previous steps"""
        try:
            if not isinstance(value, str) or not value.startswith("RESULT_FROM_STEP_"):
                return value
            
            # Parse step number
            import re
            match = re.search(r'RESULT_FROM_STEP_(\d+)', value)
            if not match:
                raise ValueError(f"Invalid placeholder format: {value}")
            
            step_num = int(match.group(1))
            result_key = f"step_{step_num}"
            
            if result_key not in context:
                raise ValueError(f"Step {step_num} not found in context")
            
            result = context[result_key]
            
            # Extract path after step reference
            path = value.replace(f"RESULT_FROM_STEP_{step_num}", "").strip()
            
            # Navigate through result
            if path:
                # Remove leading dot/bracket
                path = path.lstrip('.').lstrip('[')
                
                # Split by dots and brackets
                parts = re.split(r'[\.\[\]]', path)
                parts = [p for p in parts if p]  # Remove empty strings
                
                for part in parts:
                    if part.isdigit():
                        result = result[int(part)]
                    elif part:
                        result = result.get(part) if isinstance(result, dict) else getattr(result, part, None)
                    
                    if result is None:
                        raise ValueError(f"Path '{part}' not found in result")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to resolve '{value}': {e}")
            raise
    
    def execute_mission(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main mission execution (SYNC version)"""
        start_time = time.time()
        
        try:
            mission = mission_data.get("mission")
            if not mission:
                raise ValidationError("Mission field is required")
            
            mission = self.validator.validate_mission(mission)
            logger.info(f"📋 Mission: '{mission[:100]}...'")
            
            # Step 1: Get Plan
            architect_prompt = """You are "Architect", an AI planner for an intelligence agent.
Convert missions into JSON action plans.

Available tools:
1. semantic_search(query: str, k: int=3) - Search documents
2. get_document_content(doc_id: int) - Get full document
3. summarize_with_maverick(text: str) - Summarize text

Output ONLY a JSON array with NO other text.
Use placeholders like "RESULT_FROM_STEP_1.results[0].doc_id" for dependencies.

Example:
[
  {
    "tool_name": "semantic_search",
    "parameters": {"query": "LockBit ransomware", "k": 3},
    "description": "Search for LockBit documents"
  }
]"""
            
            logger.info("🏛️ Requesting plan...")
            
            response = self.cerebras_client.call(
                messages=[
                    {"role": "system", "content": architect_prompt},
                    {"role": "user", "content": f"Mission: {mission}"}
                ],
                model="qwen-3-235b-a22b-instruct-2507",
                temperature=0.1,
                max_tokens=2000
            )
            
            plan_str = response.choices[0].message.content
            plan = self._parse_plan_from_architect(plan_str)
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
                description = step.get("description", "")
                
                logger.info(f"▶️ Step {i+1}/{len(plan)}: {tool_name}")
                
                # Resolve parameters
                resolved_params = {}
                for key, value in parameters.items():
                    try:
                        resolved_params[key] = self._resolve_placeholder(value, context)
                    except Exception as e:
                        logger.error(f"Parameter resolution failed for '{key}': {e}")
                        resolved_params[key] = None
                
                # Execute
                if tool_name not in available_tools:
                    result = {"status": "error", "detail": f"Tool '{tool_name}' not found"}
                else:
                    try:
                        tool_func = available_tools[tool_name]
                        result = tool_func(**resolved_params)
                        logger.info(f"✅ Step {i+1} completed")
                    except Exception as e:
                        result = {"status": "error", "detail": str(e)}
                        logger.error(f"❌ Step {i+1} failed: {e}")
                
                context[f"step_{i+1}"] = result
                
                execution_log.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "description": description,
                    "parameters": resolved_params,
                    "result": result,
                    "duration_seconds": round(time.time() - step_start, 2)
                })
            
            # Step 3: Generate Report
            logger.info("📊 Generating report...")
            
            report_prompt = f"""You are an intelligence analyst. Write a comprehensive report for: '{mission}'.

Execution Log:
{json.dumps(execution_log, indent=2)}

Write in Markdown with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions

Be professional and analytical."""
            
            response = self.cerebras_client.call(
                messages=[{"role": "user", "content": report_prompt}],
                model="qwen-3-235b-thinking-2507",
                temperature=0.5,
                max_tokens=3000
            )
            
            final_report = response.choices[0].message.content
            total_duration = time.time() - start_time
            
            logger.info(f"✅ Mission completed in {total_duration:.2f}s")
            
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
            
        except ValidationError as e:
            logger.error(f"❌ Validation error: {e}")
            return {
                "success": False,
                "status": "error",
                "error": str(e),
                "mission": mission_data.get("mission", ""),
                "final_report": f"## Mission Failed\n\n**Error:** {str(e)}",
                "execution_log": []
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
