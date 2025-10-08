# 🤖 Intelligence Agent System v2.0

> **Autonomous AI-Powered Intelligence Analysis Platform**  
> Multi-step planning, semantic search, and automated report generation powered by cutting-edge AI

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/yourusername/intelligence-agent)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Modal](https://img.shields.io/badge/modal-0.63+-orange.svg)](https://modal.com/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production-success.svg)]()

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo & Contacts](#-live-demo--contacts)
- [Key Features](#-key-features)
- [Available Tools](#-available-tools)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
  - [Web Interface](#1-web-interface)
  - [Telegram Bot](#2-telegram-bot)
  - [REST API](#3-rest-api)
  - [Python SDK](#4-python-sdk)
  - [CLI Tools](#5-cli-tools)
- [Tools Deep Dive](#-tools-deep-dive)
- [Examples](#-examples)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [Support](#-support)

---

## 🎯 Overview

**Intelligence Agent System** adalah platform analisis intelijen berbasis AI yang sepenuhnya autonomous. Sistem ini mampu:

- 🧠 **Merencanakan** strategi investigasi multi-step secara otomatis
- 🔍 **Mengeksekusi** pencarian semantic dan analisis dokumen
- 📊 **Menghasilkan** laporan komprehensif dalam format profesional
- 🚀 **Menskalakan** untuk handle multiple concurrent requests

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **AI Reasoning** | Cerebras Cloud SDK | Ultra-fast inference untuk planning & analysis |
| **Embeddings** | Sentence Transformers | Semantic search capabilities |
| **Vector DB** | FAISS | Efficient similarity search |
| **Backend** | FastAPI | Modern async API framework |
| **Deployment** | Modal | Serverless container platform |
| **Bot** | python-telegram-bot | Telegram integration |

---

## 🌐 Live Demo & Contacts

### 🤖 Try the Telegram Bot
**Bot Username:** [@Durove69bot](https://t.me/Durove69bot)

Simply send a message to start investigating:
```
Investigate LockBit ransomware activities
```

### 👤 Contact Developer
**Telegram:** [@Durov9369](https://t.me/Durov9369)

For:
- 🐛 Bug reports
- 💡 Feature requests
- 🤝 Collaboration opportunities
- ❓ Technical support

### 🌐 Web Interface
Access the web UI at your Modal deployment URL:
```
https://your-username--intelligence-agent-system-fastapi-app.modal.run
```

---

## ✨ Key Features

### 🎯 Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Autonomous Planning** | AI Architect plans investigation strategies | ✅ Production |
| **Semantic Search** | Find documents by meaning, not just keywords | ✅ Production |
| **Multi-Step Execution** | Execute complex research workflows | ✅ Production |
| **Auto Summarization** | Condense long documents intelligently | ✅ Production |
| **Report Generation** | Create structured Markdown reports | ✅ Production |
| **Real-time Logging** | Track every step of execution | ✅ Production |
| **Document Management** | Add/search knowledge base | ✅ Production |
| **Rate Limiting** | Fair usage enforcement | ✅ Production |
| **Error Resilience** | Automatic retry with fallback | ✅ Production |
| **Multi-Interface** | Web, Telegram, API, CLI | ✅ Production |

### 🔒 Production-Ready

- ✅ Comprehensive error handling
- ✅ Structured logging & monitoring
- ✅ Thread-safe operations
- ✅ Resource management
- ✅ Input validation
- ✅ API authentication ready
- ✅ Health checks & metrics
- ✅ Horizontal scaling support

---

## 🛠️ Available Tools

The Intelligence Agent has access to **3 core tools** that can be chained together:

### 1️⃣ **semantic_search**

**Purpose:** Search for relevant documents using semantic understanding

**Signature:**
```python
semantic_search(query: str, k: int = 3) -> Dict[str, Any]
```

**Parameters:**
- `query` (str): Search query in natural language
- `k` (int): Number of results to return (1-10, default: 3)

**Returns:**
```json
{
  "status": "success",
  "results": [
    {
      "doc_id": 0,
      "title": "LockBit Ransomware Analysis",
      "url": "https://example.com/lockbit",
      "score": 0.89,
      "snippet": "LockBit is a ransomware-as-a-service..."
    }
  ],
  "total_found": 3
}
```

**Example Usage:**
```python
# Find documents about ransomware
result = agent.semantic_search("ransomware attack techniques", k=5)
```

**Use Cases:**
- 🔍 Threat intelligence gathering
- 📚 Research document discovery
- 🎯 Targeted information retrieval

---

### 2️⃣ **get_document_content**

**Purpose:** Retrieve full content of a specific document

**Signature:**
```python
get_document_content(doc_id: int) -> Dict[str, Any]
```

**Parameters:**
- `doc_id` (int): Document ID from search results

**Returns:**
```json
{
  "status": "success",
  "doc_id": 0,
  "title": "LockBit Ransomware Analysis",
  "url": "https://example.com/lockbit",
  "content": "Full document text here...",
  "length": 5420
}
```

**Example Usage:**
```python
# Get full content of document #5
result = agent.get_document_content(5)
full_text = result['content']
```

**Use Cases:**
- 📄 Deep document analysis
- 🔬 Detailed information extraction
- 📊 Content processing

---

### 3️⃣ **summarize_with_maverick**

**Purpose:** Summarize long text using AI (Llama Maverick)

**Signature:**
```python
summarize_with_maverick(text: str) -> str
```

**Parameters:**
- `text` (str): Text to summarize (max 8000 chars)

**Returns:**
```python
"Concise one-paragraph summary of the input text..."
```

**Example Usage:**
```python
# Summarize a long document
long_doc = agent.get_document_content(5)['content']
summary = agent.summarize_with_maverick(long_doc)
```

**Use Cases:**
- 📝 Executive summaries
- 🎯 Quick information extraction
- 📊 Report preparation

---

### 🔄 Tool Chaining

The AI Architect automatically chains tools together. Example plan:

```json
[
  {
    "tool_name": "semantic_search",
    "parameters": {"query": "LockBit ransomware", "k": 3},
    "description": "Search for LockBit documents"
  },
  {
    "tool_name": "get_document_content",
    "parameters": {"doc_id": "RESULT_FROM_STEP_1.results[0].doc_id"},
    "description": "Get full content of top result"
  },
  {
    "tool_name": "summarize_with_maverick",
    "parameters": {"text": "RESULT_FROM_STEP_2.content"},
    "description": "Summarize the document"
  }
]
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Web UI      │  │ Telegram Bot │  │  REST API    │      │
│  │  (Browser)   │  │ (@Durove69bot)│  │  (cURL/SDK)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │       FastAPI Application              │
          │     (Modal Serverless Deployment)      │
          └──────────────────┬─────────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │     Intelligence System Core           │
          │                                        │
          │  ┌────────────────────────────────┐   │
          │  │  1. AI Architect (Planner)     │   │
          │  │     Model: Qwen 235B           │   │
          │  │     Task: Create action plan   │   │
          │  └────────────────────────────────┘   │
          │                                        │
          │  ┌────────────────────────────────┐   │
          │  │  2. Tool Executor              │   │
          │  │     • semantic_search()        │   │
          │  │     • get_document_content()   │   │
          │  │     • summarize_with_maverick()│   │
          │  └────────────────────────────────┘   │
          │                                        │
          │  ┌────────────────────────────────┐   │
          │  │  3. Report Analyst             │   │
          │  │     Model: Qwen Thinking       │   │
          │  │     Task: Generate report      │   │
          │  └────────────────────────────────┘   │
          └──────────────────┬─────────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │         Data Layer                     │
          │  ┌─────────────────┐ ┌──────────────┐ │
          │  │  FAISS Index    │ │ Shelve DB    │ │
          │  │  (Vectors)      │ │ (Metadata)   │ │
          │  │  384-dim        │ │ Text content │ │
          │  └─────────────────┘ └──────────────┘ │
          └────────────────────────────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │     External Services                  │
          │  • Cerebras AI API                     │
          │  • Sentence Transformers Model         │
          └────────────────────────────────────────┘
```

### Execution Flow

```
User Input → FastAPI → Intelligence System
                ↓
           1. PLANNING PHASE
              - AI Architect analyzes mission
              - Creates multi-step execution plan
              - Determines tool sequence
                ↓
           2. EXECUTION PHASE
              For each step:
                - Resolve parameter placeholders
                - Call appropriate tool
                - Store results in context
                - Log execution details
                ↓
           3. REPORTING PHASE
              - AI Analyst reviews all results
              - Generates structured report
              - Returns to user
                ↓
           Final Report + Execution Log
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Modal account (free tier available)
- Cerebras API key ([Get here](https://cloud.cerebras.ai))

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/intelligence-agent.git
cd intelligence-agent

# Install dependencies
pip install -r requirements.txt

# Install Modal CLI
pip install modal

# Authenticate Modal
modal token new
```

### 2. Setup Secrets

```bash
# Create Cerebras API secret
modal secret create cerebras-api-key \
  CEREBRAS_API_KEY="sk-your-key-here"
```

### 3. Deploy

```bash
# Deploy to Modal
modal deploy app.py

# Note the deployment URL from output
# Example: https://username--intelligence-agent-system-fastapi-app.modal.run
```

### 4. Test

```bash
# Test health endpoint
curl https://your-url.modal.run/health

# Test mission execution
curl -X POST https://your-url.modal.run/api/execute_mission \
  -H "Content-Type: application/json" \
  -d '{"mission": "Search for information about ransomware"}'
```

### 5. Add Documents

```bash
# Add sample documents
python add_documents.py \
  --api-url https://your-url.modal.run \
  --sample
```

✅ **Done! Your Intelligence Agent is now live!**

---

## 📖 Usage Guide

## 1. Web Interface

### Access
Open browser to: `https://your-url.modal.run`

### Features
- 🎨 **Modern UI** - Clean, responsive design
- 📊 **Real-time Status** - System health monitoring
- 🔄 **Live Progress** - See execution in real-time
- 📝 **Formatted Reports** - Beautiful Markdown rendering
- 📋 **Execution Logs** - Detailed step-by-step tracking

### How to Use

1. **Enter Mission**: Type your investigation objective
   ```
   Example: Investigate APT29 Cozy Bear activities in 2024
   ```

2. **Execute**: Click "🚀 Execute Mission"

3. **View Report**: Report appears with:
   - Executive Summary
   - Key Findings
   - Detailed Analysis
   - Conclusions

4. **Check Logs**: Click execution log to see what the AI did

### Keyboard Shortcuts
- `Ctrl + Enter` - Submit mission
- `Esc` - Clear form

---

## 2. Telegram Bot

### Setup Bot

1. **Find the Bot**: [@Durove69bot](https://t.me/Durove69bot)
2. **Start**: Send `/start` command
3. **Execute Mission**: Just type your mission!

### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Welcome message & help | `/start` |
| `/help` | Detailed usage guide | `/help` |
| `/mission <text>` | Execute investigation | `/mission Analyze DDoS attacks` |
| `/status` | System health check | `/status` |
| `/stats` | Your usage statistics | `/stats` |

### Quick Mission (Recommended)

Just send your mission directly without any command:

```
Investigate LockBit ransomware TTPs
```

Bot will automatically:
1. 🎯 Understand your objective
2. 🔄 Execute investigation
3. 📊 Return formatted report
4. 📋 Provide execution log

### Features

- ✅ **No command needed** - Just send your mission
- ✅ **Inline buttons** - Interactive UI
- ✅ **Rate limiting** - 1 mission per minute
- ✅ **Execution logs** - View what the AI did
- ✅ **Long reports** - Auto-split into multiple messages
- ✅ **Cancel missions** - Stop anytime
- ✅ **Error handling** - Retry on failures

### Examples

**Threat Intelligence:**
```
Investigate APT29 Cozy Bear recent campaigns
```

**Vulnerability Research:**
```
What are the latest critical CVEs in Apache?
```

**Attack Analysis:**
```
Analyze DDoS attacks on financial sector 2024
```

---

## 3. REST API

### Base URL
```
https://your-url.modal.run
```

### Authentication
Currently no authentication required. For production, implement API key auth.

### Endpoints

#### POST `/api/execute_mission`

Execute intelligence mission.

**Request:**
```bash
curl -X POST https://your-url.modal.run/api/execute_mission \
  -H "Content-Type: application/json" \
  -d '{
    "mission": "Investigate ransomware attacks on healthcare"
  }'
```

**Response:**
```json
{
  "success": true,
  "status": "success",
  "mission": "Investigate ransomware attacks on healthcare",
  "final_report": "## Executive Summary\n\n...",
  "execution_log": [
    {
      "step": 1,
      "tool": "semantic_search",
      "description": "Search for ransomware documents",
      "parameters": {"query": "ransomware healthcare", "k": 3},
      "result": {...},
      "duration_seconds": 1.23
    }
  ],
  "metadata": {
    "total_steps": 3,
    "successful_steps": 3,
    "total_duration_seconds": 5.67
  }
}
```

#### GET `/health`

Check system health.

**Request:**
```bash
curl https://your-url.modal.run/health
```

**Response:**
```json
{
  "status": "healthy",
  "uptime": 3600.5,
  "documents": 150,
  "cache_size": 1024000,
  "timestamp": 1699900000.0
}
```

#### GET `/metrics`

Get system metrics.

**Request:**
```bash
curl https://your-url.modal.run/metrics
```

**Response:**
```json
{
  "timestamp": 1699900000.0,
  "system": {
    "status": "healthy",
    "uptime": 3600.5,
    "documents": 150,
    "cache_size": 1024000
  },
  "version": "2.0.0"
}
```

#### POST `/api/add_document`

Add document to knowledge base.

**Request:**
```bash
curl -X POST https://your-url.modal.run/api/add_document \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document content here...",
    "url": "https://source-url.com",
    "title": "Document Title"
  }'
```

**Response:**
```json
{
  "status": "success",
  "doc_id": 42,
  "message": "Document added with ID 42"
}
```

---

## 4. Python SDK

### Usage

```python
import requests

class IntelligenceAgentClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
    
    def execute_mission(self, mission: str) -> dict:
        """Execute intelligence mission"""
        response = requests.post(
            f"{self.base_url}/api/execute_mission",
            json={"mission": mission},
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    
    def add_document(self, text: str, url: str = "", title: str = "") -> dict:
        """Add document to knowledge base"""
        response = requests.post(
            f"{self.base_url}/api/add_document",
            json={"text": text, "url": url, "title": title}
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        """Check system health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Initialize client
client = IntelligenceAgentClient("https://your-url.modal.run")

# Execute mission
result = client.execute_mission(
    "Investigate LockBit ransomware activities"
)

print(result['final_report'])

# Add document
client.add_document(
    text="APT29 is a sophisticated threat actor...",
    url="https://example.com/apt29",
    title="APT29 Analysis"
)

# Health check
health = client.health_check()
print(f"System status: {health['status']}")
print(f"Documents: {health['documents']}")
```

---

## 5. CLI Tools

### add_documents.py

Bulk import tool for adding documents.

**Features:**
- ✅ JSON file import
- ✅ Sample document templates
- ✅ Progress tracking
- ✅ Error handling
- ✅ Summary statistics

**Usage:**

```bash
# Add sample documents
python add_documents.py \
  --api-url https://your-url.modal.run \
  --sample

# Import from JSON file
python add_documents.py \
  --api-url https://your-url.modal.run \
  --json documents.json

# Create sample JSON template
python add_documents.py --create-sample

# Custom delay between requests
python add_documents.py \
  --api-url https://your-url.modal.run \
  --json documents.json \
  --delay 2.0
```

**JSON Format:**

```json
[
  {
    "title": "Document Title",
    "url": "https://source-url.com",
    "text": "Full document content here..."
  }
]
```

---

## 🔬 Tools Deep Dive

### Semantic Search Implementation

The semantic search uses a sophisticated pipeline:

```python
# 1. Text Encoding
user_query = "ransomware attacks"
query_vector = sentence_transformer.encode(user_query)
# Output: 384-dimensional vector

# 2. Vector Similarity Search
distances, indices = faiss_index.search(query_vector, k=3)
# Uses L2 distance for similarity

# 3. Result Ranking
scores = [1 - distance for distance in distances[0]]
# Convert distance to similarity score (0-1)

# 4. Metadata Retrieval
results = []
for idx, score in zip(indices[0], scores):
    doc_metadata = shelve_db[str(idx)]
    results.append({
        "doc_id": idx,
        "score": score,
        "title": doc_metadata['title'],
        ...
    })
```

### AI Planning Mechanism

The Architect uses few-shot learning:

```python
system_prompt = """
You are "Architect", an AI planner for investigations.

Available tools:
1. semantic_search(query, k) - Find relevant documents
2. get_document_content(doc_id) - Get full document
3. summarize_with_maverick(text) - Summarize text

Output a JSON plan with steps to achieve the mission.
Use placeholders for dependent parameters.

Example:
Mission: "Research LockBit ransomware"

Plan:
[
  {
    "tool_name": "semantic_search",
    "parameters": {"query": "LockBit ransomware", "k": 3},
    "description": "Find relevant documents"
  },
  {
    "tool_name": "get_document_content",
    "parameters": {"doc_id": "RESULT_FROM_STEP_1.results[0].doc_id"},
    "description": "Get top result content"
  }
]
"""
```

### Report Generation

The Analyst synthesizes findings:

```python
report_prompt = f"""
Analyze execution log and create report for: '{mission}'

Execution Log:
{json.dumps(execution_log, indent=2)}

Structure:
1. Executive Summary (3-4 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (2-3 paragraphs)
4. Conclusions

Use professional, analytical tone.
"""

# Model: Qwen Thinking (with chain-of-thought)
report = cerebras.chat.completions.create(
    model="qwen-3-235b-thinking-2507",
    messages=[{"role": "user", "content": report_prompt}],
    temperature=0.5
)
```

---

## 💡 Examples

### Example 1: Threat Intelligence Research

**Mission:**
```
Investigate APT29 Cozy Bear recent activities and TTPs
```

**AI Plan Generated:**
```json
[
  {
    "step": 1,
    "tool": "semantic_search",
    "parameters": {"query": "APT29 Cozy Bear TTPs 2024", "k": 5}
  },
  {
    "step": 2,
    "tool": "get_document_content",
    "parameters": {"doc_id": "RESULT_FROM_STEP_1.results[0].doc_id"}
  },
  {
    "step": 3,
    "tool": "summarize_with_maverick",
    "parameters": {"text": "RESULT_FROM_STEP_2.content"}
  }
]
```

**Report Output:**
```markdown
## Executive Summary

APT29 (Cozy Bear) has demonstrated increased sophistication in 2024,
with focus on supply chain compromises and cloud infrastructure targeting...

## Key Findings

- Enhanced use of legitimate cloud services for C2
- Shift to stealthier, longer-term operations
- Targeting of diplomatic and think tank entities

## Detailed Analysis

Recent campaigns show APT29 leveraging zero-day vulnerabilities...

## Conclusions

Organizations should implement enhanced monitoring...
```

---

### Example 2: Vulnerability Assessment

**Mission:**
```
What are the latest critical vulnerabilities in Apache products?
```

**Execution Flow:**
1. Search for "Apache critical vulnerabilities CVE"
2. Retrieve top 3 matching documents
3. Summarize each document
4. Generate comparative analysis

**Report Snippet:**
```markdown
## Latest Apache Critical Vulnerabilities

### CVE-2023-XXXXX (Apache HTTP Server)
Severity: Critical (CVSS 9.8)
Impact: Remote Code Execution...

### CVE-2023-YYYYY (Apache Log4j)
Severity: High (CVSS 8.1)
Impact: Information Disclosure...
```

---

### Example 3: Ransomware Campaign Analysis

**Mission:**
```
Analyze LockBit 3.0 ransomware tactics and recent campaigns
```

**Tools Used:**
1. `semantic_search("LockBit 3.0 ransomware tactics", k=5)`
2. `get_document_content(top_3_results)`
3. `summarize_with_maverick(combined_content)`

**Report Includes:**
- Infection vectors
- Encryption methods
- Ransom demands
- Attribution
- Mitigation strategies

---

## 📦 Deployment

### Modal Deployment (Recommended)

**Advantages:**
- ✅ Serverless - pay only for usage
- ✅ Auto-scaling
- ✅ Built-in secrets management
- ✅ Easy deployment
- ✅ Container isolation

**Steps:**
1. Install Modal: `pip install modal`
2. Authenticate: `modal token new`
3. Create secrets: `modal secret create cerebras-api-key ...`
4. Deploy: `modal deploy app.py`
5. Done! Get URL from output

**Costs:**
- Free tier: 30 free credits/month
- Pay-as-you-go after free tier
- Typical mission: ~$0.01-0.05

### Alternative: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV CEREBRAS_API_KEY=your-key

CMD ["uvicorn", "app:web_app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
docker build -t intelligence-agent .
docker run -p 8000:8000 -e CEREBRAS_API_KEY=your-key intelligence-agent
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Required
CEREBRAS_API_KEY=sk-xxxxx          # Primary API key
CEREBRAS_API_KEY_2=sk-yyyyy        # Backup key (optional)

# Optional
CACHE_DIR=/cache                   # Cache directory
VECTOR_DIMENSION=384               # Embedding dimension
MODEL_NAME=all-MiniLM-L6-v2       # Sentence transformer model
```

### Modal Configuration

Edit `app.py`:

```python
@app.function(
    cpu=2.0,                       # CPU cores (0.1-16)
    memory=4096,                   # RAM in MB (128-65536)
    timeout=900,                   # Max execution time in seconds
    keep_warm=1,                   # Containers to keep warm
    container_idle_timeout=300     # Idle timeout in seconds
)
```

### Telegram Bot Configuration

Edit `telegram_bot.py`:

```python
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
API_BASE_URL = "https://your-url.modal.run"
RATE_LIMIT_SECONDS = 60           # Rate limit per user
ADMIN_USER_IDS = [123456789]      # Admin user IDs
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/intelligence-agent.git
cd intelligence-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # pytest, black, etc.

# Run tests
pytest tests/

# Format code
black .
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Code Standards

- ✅ Follow PEP 8
- ✅ Add type hints
- ✅ Write docstrings
- ✅ Include tests
- ✅ Update documentation

---

## 📞 Support

### 🐛 Found a Bug?

1. **Check** [existing issues](https://github.com/yourusername/intelligence-agent/issues)
2. **Create** new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable

### 💡 Feature Request?

Open an issue with:
- Feature description
- Use case
- Why it's valuable

### 🤝 Need Help?

- 💬 **Telegram:** [@Durov9369](https://t.me/Durov9369)
- 📧 **Email:** support@yourproject.com
- 📖 **Docs:** [Read full documentation](https://docs.yourproject.com)
- 🤖 **Test Bot:** [@Durove69bot](https://t.me/Durove69bot)

### 📚 Resources

- [Modal Documentation](https://modal.com/docs)
- [Cerebras AI Docs](https://inference-docs.cerebras.ai/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Cerebras** - Ultra-fast AI inference
- **Sentence Transformers** - Semantic embeddings
- **FAISS** - Efficient vector search
- **Modal** - Serverless deployment
- **FastAPI** - Modern web framework
- **python-telegram-bot** - Telegram integration

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

## 📊 Project Stats

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Code Size](https://img.shields.io/github/languages/code-size/yourusername/intelligence-agent)
![Contributors](https://img.shields.io/github/contributors/yourusername/intelligence-agent)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/intelligence-agent)
![Issues](https://img.shields.io/github/issues/yourusername/intelligence-agent)
![Pull Requests](https://img.shields.io/github/issues-pr/yourusername/intelligence-agent)

---

<div align="center">

**Made with ❤️ by the Intelligence Agent Team**

[Website](https://yourproject.com) • 
[Documentation](https://docs.yourproject.com) • 
[Telegram Bot](https://t.me/Durove69bot) • 
[Contact Developer](https://t.me/Durov9369)

⭐ **Star us on GitHub!** ⭐

</div>
