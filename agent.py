# agent.py (Production-Ready Refactored Version)

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
        
        from sentence_transformers import SentenceTransformer
        
        logger.info("🔧 Initializing embedding model (singleton)...")
        try:
            self.model = SentenceTransformer(MODEL_NAME, cache_folder=CACHE_DIR)
            self.dimension = VECTOR_DIMENSION
            self._initialized = True
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    @retry_with_exponential_backoff(max_retries=3)
    def encode(self, texts: List[str]) -> Any:
        """Encode texts dengan retry mechanism"""
        try:
            return self.model.encode(texts)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise


class VectorStoreManager:
    """Mengelola FAISS index dengan thread-safety"""
    
    def __init__(self):
        import faiss
        
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
        self.api_keys = [key for key in self.api_keys if key]  # Filter None values
        
        if not self.api_keys:
            raise ValueError("❌ No valid Cerebras API keys found in environment")
        
        logger.info(f"🔑 Initialized with {len(self.api_keys)} API key(s)")
    
    @retry_with_exponential_backoff(max_retries=2, initial_delay=0.5)
    def call(self, **kwargs) -> Any:
        """Call Cerebras API dengan automatic fallback"""
        from cerebras.cloud.sdk import Cerebras
        from cerebras.cloud.sdk.errors import AuthenticationError, APIError
        
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
                # Don't try other keys for API errors (rate limits, etc)
                raise
                
            except Exception as e:
                logger.error(f"❌ Unexpected error with key #{i+1}: {str(e)[:100]}")
                last_error = e
                raise
        
        # If we've exhausted all keys
        raise AuthenticationError(f"All {len(self.api_keys)} API key(s) failed. Last error: {last_error}")


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
    
    # --- Tools ---
    
    def semantic_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Semantic search dengan comprehensive error handling dan validation.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            Dict containing status and results/error details
        """
        try:
            # Validation
            query = self.validator.validate_query(query)
            
            if k < 1 or k > 10:
                k = min(max(k, 1), 10)
                logger.warning(f"k adjusted to valid range: {k}")
            
            # Check if database is empty
            if self.vector_store.get_total_vectors() == 0:
                return {
                    "status": "error",
                    "detail": "Database is empty. No documents to search.",
                    "results": []
                }
            
            # Encode query
            query_vector = self.embedding_manager.encode([query])[0]
            
            # Search
            distances, ids = self.vector_store.search(query_vector, k)
            
            # Retrieve document metadata
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
            
            logger.info(f"✅ Search completed: {len(results)} results for query: '{query[:50]}...'")
            
            return {
                "status": "success",
                "results": results,
                "total_found": len(results)
            }
            
        except ValidationError as e:
            logger.warning(f"Validation error in semantic_search: {e}")
            return {"status": "error", "detail": str(e), "results": []}
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "detail": f"Search failed: {str(e)}",
                "results": []
            }
    
    def get_document_content(self, doc_id: int) -> Dict[str, Any]:
        """
        Retrieve full document content dengan validation dan error handling.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Dict containing status and content/error details
        """
        try:
            # Validation
            doc_id = self.validator.validate_doc_id(doc_id)
            
            # Retrieve document
            with safe_shelve_open(DOC_STORE_PATH) as doc_store:
                doc_key = str(doc_id)
                
                if doc_key not in doc_store:
                    logger.warning(f"Document {doc_id} not found")
                    return {
                        "status": "error",
                        "detail": f"Document with ID {doc_id} not found in database"
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
            logger.warning(f"Validation error in get_document_content: {e}")
            return {"status": "error", "detail": str(e)}
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve document {doc_id}: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "detail": f"Failed to retrieve document: {str(e)}"
            }
    
    def summarize_with_maverick(self, text: str) -> str:
        """
        Summarize text using Llama Maverick dengan error handling.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summary string
        """
        try:
            if not text or not isinstance(text, str):
                raise ValidationError("Text must be a non-empty string")
            
            # Truncate if too long
            max_length = 8000
            if len(text) > max_length:
                text = text[:max_length] + "..."
                logger.warning(f"Text truncated to {max_length} characters for summarization")
            
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
            logger.info(f"✅ Summarization completed ({len(text)} -> {len(summary)} chars)")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}\n{traceback.format_exc()}")
            return f"[Summarization failed: {str(e)}]"
    
    # --- Mission Execution ---
    
    def _parse_plan_from_architect(self, plan_str: str) -> List[Dict[str, Any]]:
        """Parse and validate plan from architect"""
        try:
            # Remove markdown code blocks if present
            if plan_str.strip().startswith("```"):
                lines = plan_str.strip().split('\n')
                plan_str = '\n'.join(lines[1:-1]) if len(lines) > 2 else plan_str
            
            plan = json.loads(plan_str)
            
            # Validate plan structure
            if not isinstance(plan, list):
                raise ValidationError("Plan must be a JSON array")
            
            for i, step in enumerate(plan):
                if not isinstance(step, dict):
                    raise ValidationError(f"Step {i+1} must be a JSON object")
                
                if "tool_name" not in step:
                    raise ValidationError(f"Step {i+1} missing 'tool_name' field")
            
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise ValidationError(f"Invalid JSON format: {str(e)}")
    
    def _resolve_placeholder(self, value: str, context: Dict[str, Any]) -> Any:
        """Resolve placeholder references dari previous steps"""
        try:
            if not value.startswith("RESULT_FROM_STEP_"):
                return value
            
            # Parse: RESULT_FROM_STEP_1[0].doc_id
            parts = value.replace("]", "").replace("[", ".").split('.')
            step_num = int(parts[0].split('_')[-1])
            result_key = f"step_{step_num}"
            
            if result_key not in context:
                raise ValueError(f"Step {step_num} not found in context")
            
            result = context[result_key]
            
            # Navigate through the result object
            for part in parts[1:]:
                if part.isdigit():
                    result = result[int(part)]
                elif part:
                    result = result[part]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to resolve placeholder '{value}': {e}")
            raise
    
    async def execute_mission(self, mission_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main mission execution dengan comprehensive error handling dan logging.
        
        Args:
            mission_data: Dict containing 'mission' field
            
        Returns:
            Dict containing mission results, report, and execution log
        """
        start_time = time.time()
        
        try:
            # --- Input Validation ---
            mission = mission_data.get("mission")
            if not mission:
                raise ValidationError("Mission field is required and cannot be empty")
            
            mission = self.validator.validate_mission(mission)
            logger.info(f"📋 Mission received: '{mission[:100]}...'")
            
            # --- Step 1: Get Plan from Architect ---
            architect_system_prompt = """
You are "Architect", an autonomous AI planner for an intelligence agent.
Your task is to receive a Mission and convert it into an Action Plan in JSON format.
This plan is a list of steps to be executed sequentially.

You have access to these tools:
1. semantic_search(query: str, k: int=3): Search for relevant documents. Use this to start investigations.
2. get_document_content(doc_id: int): Retrieve full text of a found document.
3. summarize_with_maverick(text: str): Summarize long text for easier understanding.

Your output MUST be a JSON array-of-objects, with no other explanatory text.
Use placeholders for parameters that depend on previous step results.
Example: {"doc_id": "RESULT_FROM_STEP_1.results[0].doc_id"}

Each step object must have:
- "tool_name": Name of the tool to use
- "parameters": Object with tool parameters
- "description": Brief description of what this step does

Example output:
[
  {
    "tool_name": "semantic_search",
    "parameters": {"query": "ransomware LockBit activities", "k": 3},
    "description": "Search for documents about LockBit ransomware"
  },
  {
    "tool_name": "get_document_content",
    "parameters": {"doc_id": "RESULT_FROM_STEP_1.results[0].doc_id"},
    "description": "Retrieve full content of most relevant document"
  }
]
"""
            
            logger.info("🏛️ Requesting plan from Architect...")
            
            response = self.cerebras_client.call(
                messages=[
                    {"role": "system", "content": architect_system_prompt},
                    {"role": "user", "content": f"Mission: {mission}"}
                ],
                model="qwen-3-235b-a22b-instruct-2507",
                temperature=0.1,
                max_tokens=2000
            )
            
            plan_str = response.choices[0].message.content
            logger.info(f"📜 Raw plan received: {plan_str[:200]}...")
            
            # Parse and validate plan
            plan = self._parse_plan_from_architect(plan_str)
            logger.info(f"🗺️ Plan validated: {len(plan)} steps")
            
            # --- Step 2: Execute Plan ---
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
                description = step.get("description", "No description")
                
                logger.info(f"▶️ Step {i+1}/{len(plan)}: {tool_name} - {description}")
                
                # Resolve placeholders in parameters
                resolved_params = {}
                for key, value in parameters.items():
                    try:
                        if isinstance(value, str) and "RESULT_FROM_STEP_" in value:
                            resolved_params[key] = self._resolve_placeholder(value, context)
                        else:
                            resolved_params[key] = value
                    except Exception as e:
                        logger.error(f"Failed to resolve parameter '{key}': {e}")
                        resolved_params[key] = None
                
                # Execute tool
                if tool_name not in available_tools:
                    result = {
                        "status": "error",
                        "detail": f"Tool '{tool_name}' not found in toolbox"
                    }
                    logger.error(f"❌ {result['detail']}")
                else:
                    try:
                        tool_function = available_tools[tool_name]
                        result = tool_function(**resolved_params)
                        logger.info(f"✅ Step {i+1} completed in {time.time() - step_start:.2f}s")
                    except Exception as e:
                        result = {
                            "status": "error",
                            "detail": f"Tool execution failed: {str(e)}"
                        }
                        logger.error(f"❌ Step {i+1} failed: {e}\n{traceback.format_exc()}")
                
                # Store in context
                context[f"step_{i+1}"] = result
                
                execution_log.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "description": description,
                    "parameters": resolved_params,
                    "result": result,
                    "duration_seconds": round(time.time() - step_start, 2)
                })
            
            # --- Step 3: Generate Final Report ---
            logger.info("📊 Generating final report...")
            
            final_report_prompt = f"""
You are an intelligence analyst. Based on the following mission execution log, 
write a comprehensive final report for the initial mission: '{mission}'.

Write in Markdown format with clear sections. Include:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions
5. Recommendations (if applicable)

Execution Log:
{json.dumps(execution_log, indent=2)}

Write the report in a professional, analytical style.
"""
            
            response = self.cerebras_client.call(
                messages=[{"role": "user", "content": final_report_prompt}],
                model="qwen-3-235b-a22b-thinking-2507",
                temperature=0.5,
                max_tokens=3000
            )
            
            final_report = response.choices[0].message.content
            
            total_duration = time.time() - start_time
            logger.info(f"✅ Mission completed in {total_duration:.2f}s")
            
            return {
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
                "status": "error",
                "error": str(e),
                "mission": mission_data.get("mission", ""),
                "final_report": f"## Mission Failed\n\n**Error:** {str(e)}\n\nPlease check your input and try again.",
                "execution_log": []
            }
            
        except Exception as e:
            logger.error(f"❌ Mission execution failed: {e}\n{traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "mission": mission_data.get("mission", ""),
                "final_report": f"## Mission Failed\n\n**Error:** An unexpected error occurred: {str(e)}\n\nPlease try again or contact support if the issue persists.",
                "execution_log": []
            }
