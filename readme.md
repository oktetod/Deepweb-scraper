# 🤖 Intelligence Agent System

> An autonomous AI-powered intelligence analysis platform with semantic search, document retrieval, and automated report generation.

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/yourusername/intelligence-agent)
[![Python](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [Web UI](#1-web-ui)
  - [API Integration](#2-api-integration)
  - [Telegram Bot Integration](#3-telegram-bot-integration)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🎯 Overview

Intelligence Agent System adalah platform analisis intelijen berbasis AI yang dapat:

- **Merencanakan** strategi investigasi secara otomatis
- **Mengeksekusi** pencarian dan analisis multi-step
- **Menghasilkan** laporan komprehensif dalam format Markdown

Sistem ini menggunakan:
- **Cerebras AI** untuk reasoning dan perencanaan
- **Sentence Transformers** untuk semantic search
- **FAISS** untuk vector similarity search
- **Modal** untuk deployment dan scaling

---

## ✨ Features

### Core Capabilities

- 🧠 **Autonomous Planning**: AI Architect merencanakan strategi investigasi
- 🔍 **Semantic Search**: Pencarian dokumen berbasis makna, bukan keyword
- 📄 **Document Retrieval**: Akses konten lengkap dari database
- 📊 **Auto Summarization**: Ringkasan otomatis untuk teks panjang
- 📝 **Report Generation**: Laporan terstruktur dengan analisis mendalam

### Technical Features

- ⚡ **High Performance**: Caching, connection pooling, lazy loading
- 🔒 **Thread-Safe**: Concurrent request handling
- 🛡️ **Error Resilient**: Comprehensive error handling & retry logic
- 📈 **Production-Ready**: Logging, monitoring, health checks
- 🌐 **RESTful API**: Easy integration dengan aplikasi lain

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Interface                    │
│            (Web UI / Telegram Bot / API)            │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  FastAPI Server                      │
│  ┌──────────────────────────────────────────────┐  │
│  │        Intelligence System (agent.py)        │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │  1. Architect (Planning)               │  │  │
│  │  │     - Cerebras Qwen 235B               │  │  │
│  │  └────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │  2. Executor (Tools)                   │  │  │
│  │  │     - semantic_search()                │  │  │
│  │  │     - get_document_content()           │  │  │
│  │  │     - summarize_with_maverick()        │  │  │
│  │  └────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │  3. Analyst (Reporting)                │  │  │
│  │  │     - Cerebras Qwen Thinking           │  │  │
│  │  └────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              Data Layer                              │
│  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  FAISS Index    │  │  Shelve Document Store  │  │
│  │  (Vectors)      │  │  (Metadata + Content)   │  │
│  └─────────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Modal account (untuk deployment)
- Cerebras API key

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/intelligence-agent.git
cd intelligence-agent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Create secrets
modal secret create cerebras-api-key \
  CEREBRAS_API_KEY="your-primary-key" \
  CEREBRAS_API_KEY_2="your-backup-key"
```

### Step 4: Deploy

```bash
# Deploy to Modal
modal deploy app.py

# Get deployment URL
modal app list
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Cerebras API Keys (stored in Modal secrets)
CEREBRAS_API_KEY=sk-xxxxxxxxxxxxxx
CEREBRAS_API_KEY_2=sk-yyyyyyyyyyyyyy  # Optional backup key
```

### System Configuration (agent.py)

```python
# Cache and storage paths
CACHE_DIR = "/cache"
DB_DIR = Path(f"{CACHE_DIR}/db")
DOC_STORE_PATH = str(DB_DIR / "documents.db")
FAISS_INDEX_PATH = str(DB_DIR / "vectors.faiss")

# Model configuration
VECTOR_DIMENSION = 384
MODEL_NAME = "all-MiniLM-L6-v2"
```

### Modal Configuration (app.py)

```python
# Resource allocation
cpu=2.0  # CPU cores
container_idle_timeout=300  # 5 minutes
keep_warm=1  # Keep 1 container warm
timeout=900  # 15 minutes max execution
```

---

## 📖 Usage

## 1. Web UI

### Accessing the Interface

Setelah deployment, buka URL Modal yang diberikan di browser:

```
https://your-app-name--final-app.modal.run
```

### Using the Web Interface

1. **Enter Mission**: Masukkan objektif investigasi di text area
   ```
   Example: "Investigate recent activities of ransomware group 'LockBit' 
   and analyze their attack patterns"
   ```

2. **Execute Mission**: Klik tombol "🚀 Execute Mission"

3. **View Results**: 
   - Report ditampilkan dalam format Markdown
   - Execution log menunjukkan detail setiap step
   - Metrics menampilkan durasi dan success rate

### Features

- ✅ Real-time status updates
- ✅ System health monitoring
- ✅ Responsive design (mobile-friendly)
- ✅ Markdown rendering
- ✅ Execution log viewer

---

## 2. API Integration

### Base URL

```
https://your-app-name--final-app.modal.run
```

### Authentication

Tidak ada authentication untuk public endpoints. Untuk production, tambahkan API key authentication.

### Endpoints

#### Execute Mission

```http
POST /api/execute_mission
Content-Type: application/json

{
  "mission": "Your mission description here"
}
```

**Example Request (curl):**

```bash
curl -X POST "https://your-app-name--final-app.modal.run/api/execute_mission" \
  -H "Content-Type: application/json" \
  -d '{
    "mission": "Analyze the latest cybersecurity threats in the financial sector"
  }'
```

**Example Request (Python):**

```python
import requests

url = "https://your-app-name--final-app.modal.run/api/execute_mission"
payload = {
    "mission": "Investigate DDoS attacks targeting government websites"
}

response = requests.post(url, json=payload)
result = response.json()

print(result["final_report"])
```

**Response Structure:**

```json
{
  "status": "success",
  "mission": "Your mission text",
  "final_report": "## Executive Summary\n\n...",
  "execution_log": [
    {
      "step": 1,
      "tool": "semantic_search",
      "description": "Search for relevant documents",
      "parameters": {"query": "...", "k": 3},
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

#### Health Check

```http
GET /health
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

#### Metrics

```http
GET /metrics
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

---

## 3. Telegram Bot Integration

### Setup

#### Step 1: Create Bot

```bash
# Talk to @BotFather on Telegram
/newbot
# Follow instructions and get your bot token
```

#### Step 2: Install Python Telegram Bot

```bash
pip install python-telegram-bot==20.0
```

#### Step 3: Create Bot Script

```python
# telegram_bot.py

import asyncio
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import requests

# Configuration
TELEGRAM_TOKEN = "your-bot-token-here"
API_URL = "https://your-app-name--final-app.modal.run/api/execute_mission"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Command Handlers

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    welcome_text = """
🤖 *Intelligence Agent Bot*

I'm an autonomous AI agent for intelligence analysis.

*Commands:*
/start - Show this help message
/mission <your mission> - Execute intelligence mission
/status - Check system status

*Example:*
`/mission Investigate recent ransomware attacks`

Just describe what you want to investigate, and I'll handle the rest!
    """
    await update.message.reply_text(
        welcome_text,
        parse_mode='Markdown'
    )

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check system status"""
    try:
        response = requests.get(f"{API_URL.replace('/api/execute_mission', '/health')}")
        data = response.json()
        
        status_text = f"""
📊 *System Status*

Status: {data['status'].upper()}
Documents: {data['documents']}
Uptime: {int(data['uptime'])}s

All systems operational ✅
        """
        await update.message.reply_text(status_text, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Error checking status: {str(e)}")

async def mission(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute intelligence mission"""
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a mission description.\n\n"
            "Example: `/mission Analyze cybersecurity threats`",
            parse_mode='Markdown'
        )
        return
    
    mission_text = ' '.join(context.args)
    
    # Send initial message
    status_message = await update.message.reply_text(
        f"🔄 *Executing Mission*\n\n"
        f"Mission: _{mission_text}_\n\n"
        f"Please wait while I analyze...",
        parse_mode='Markdown'
    )
    
    try:
        # Call API
        response = requests.post(
            API_URL,
            json={"mission": mission_text},
            timeout=300  # 5 minutes
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "success":
            # Send report in chunks if too long
            report = result["final_report"]
            metadata = result.get("metadata", {})
            
            # Add metadata footer
            footer = f"\n\n---\n⏱️ Completed in {metadata.get('total_duration_seconds', 0)}s\n"
            footer += f"✅ {metadata.get('successful_steps', 0)}/{metadata.get('total_steps', 0)} steps successful"
            
            full_message = f"{report}{footer}"
            
            # Split if too long (Telegram limit: 4096 chars)
            if len(full_message) > 4000:
                chunks = [full_message[i:i+4000] for i in range(0, len(full_message), 4000)]
                await status_message.delete()
                for chunk in chunks:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
            else:
                await status_message.edit_text(full_message, parse_mode='Markdown')
        else:
            error_msg = result.get("error", "Unknown error")
            await status_message.edit_text(
                f"❌ *Mission Failed*\n\nError: {error_msg}",
                parse_mode='Markdown'
            )
    
    except requests.exceptions.Timeout:
        await status_message.edit_text(
            "⏱️ Request timeout. Mission may still be processing. Try again later."
        )
    except Exception as e:
        logger.error(f"Mission execution error: {e}")
        await status_message.edit_text(
            f"❌ *Error*\n\n{str(e)}",
            parse_mode='Markdown'
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages as mission commands"""
    message_text = update.message.text
    
    # Treat any message as a mission
    context.args = message_text.split()
    await mission(update, context)

# Main function

def main():
    """Start the bot"""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("mission", mission))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    ))
    
    # Start bot
    logger.info("🤖 Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
```

#### Step 4: Run Bot

```bash
python telegram_bot.py
```

### Bot Usage

#### Basic Commands

```
/start              - Show help message
/status             - Check system status
/mission <text>     - Execute intelligence mission
```

#### Examples

```
/mission Investigate LockBit ransomware attacks in 2024

/mission Analyze phishing campaigns targeting healthcare sector

/mission What are the latest zero-day vulnerabilities?
```

#### Quick Mission (No Command)

Kirim pesan langsung tanpa `/mission`:

```
Investigate DDoS attacks on government websites
```

Bot akan otomatis treat sebagai mission command.

### Advanced Features

#### Add Inline Buttons

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

async def mission_with_options(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("📊 Full Report", callback_data='full'),
            InlineKeyboardButton("📝 Summary Only", callback_data='summary')
        ],
        [InlineKeyboardButton("❌ Cancel", callback_data='cancel')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Choose report type:",
        reply_markup=reply_markup
    )
```

#### User Authentication

```python
AUTHORIZED_USERS = [123456789, 987654321]  # User IDs

async def mission(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id not in AUTHORIZED_USERS:
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    # Continue with mission execution...
```

#### Rate Limiting

```python
from collections import defaultdict
import time

user_last_request = defaultdict(float)
RATE_LIMIT = 60  # seconds

async def mission(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current_time = time.time()
    
    if current_time - user_last_request[user_id] < RATE_LIMIT:
        remaining = int(RATE_LIMIT - (current_time - user_last_request[user_id]))
        await update.message.reply_text(
            f"⏳ Please wait {remaining}s before next request"
        )
        return
    
    user_last_request[user_id] = current_time
    # Continue with mission execution...
```

---

## 📚 API Reference

### Complete API Documentation

#### Error Responses

All endpoints may return error responses:

```json
{
  "error": "Error description",
  "detail": "Detailed error message",
  "timestamp": 1699900000.0
}
```

#### HTTP Status Codes

- `200 OK` - Successful request
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - System unhealthy

#### Request Validation

**Mission Requirements:**
- Minimum length: 10 characters
- Maximum length: 2000 characters
- Must be non-empty string
- Will be automatically trimmed

---

## 🛠️ Development

### Local Development

```bash
# Set environment variables
export CEREBRAS_API_KEY="your-key"

# Run with Modal dev mode
modal serve app.py

# Access at http://localhost:8000
```

### Adding New Tools

Edit `agent.py` and add to `IntelligenceSystem` class:

```python
def your_new_tool(self, param1: str, param2: int) -> Dict[str, Any]:
    """
    Your tool description
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Dict with status and results
    """
    try:
        # Your implementation
        result = do_something(param1, param2)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Tool failed: {e}")
        return {
            "status": "error",
            "detail": str(e)
        }

# Register in execute_mission():
available_tools = {
    "semantic_search": self.semantic_search,
    "get_document_content": self.get_document_content,
    "summarize_with_maverick": self.summarize_with_maverick,
    "your_new_tool": self.your_new_tool,  # Add here
}
```

### Running Tests

```bash
# Install pytest
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No valid Cerebras API keys found"

**Solution:**
```bash
# Verify secrets are set
modal secret list

# Update secret
modal secret create cerebras-api-key CEREBRAS_API_KEY="new-key"
```

#### 2. "Database is empty"

**Solution:** Populate database terlebih dahulu. Implementasi `add_document()` function:

```python
def add_document(self, text: str, url: str, title: str):
    # Generate embedding
    vector = self.embedding_manager.encode([text])[0]
    
    # Get next ID
    doc_id = self.vector_store.get_total_vectors()
    
    # Add to FAISS
    self.vector_store.index.add_with_ids(
        np.array([vector], dtype=np.float32),
        np.array([doc_id], dtype=np.int64)
    )
    
    # Save metadata
    with safe_shelve_open(DOC_STORE_PATH, flag='c') as doc_store:
        doc_store[str(doc_id)] = {
            "text": text,
            "url": url,
            "title": title
        }
    
    self.vector_store.save()
```

#### 3. "Request timeout"

**Solution:** Increase timeout di `app.py`:

```python
@modal.function(
    timeout=1800,  # 30 minutes instead of 15
    ...
)
```

#### 4. Telegram bot tidak merespons

**Checklist:**
- ✅ Token benar
- ✅ API URL correct
- ✅ Bot running (`python telegram_bot.py`)
- ✅ Network tidak diblock
- ✅ Check logs untuk error messages

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Cerebras** for ultra-fast inference
- **Sentence Transformers** for semantic embeddings
- **FAISS** for efficient vector search
- **Modal** for serverless deployment

---

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join Server](https://discord.gg/example)
- 📖 Docs: [Read Docs](https://docs.example.com)

---

**Made with ❤️ by Intelligence Team**