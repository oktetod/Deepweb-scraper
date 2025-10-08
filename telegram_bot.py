"""
Telegram Bot Integration for Intelligence Agent System
Production-ready with advanced features
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
API_BASE_URL = os.getenv("API_BASE_URL", "https://your-app--fastapi-app.modal.run")
API_EXECUTE = f"{API_BASE_URL}/api/execute_mission"
API_HEALTH = f"{API_BASE_URL}/health"
API_ADD_DOC = f"{API_BASE_URL}/api/add_document"

# Rate limiting
RATE_LIMIT_SECONDS = 60
ADMIN_USER_IDS = []  # Add admin Telegram user IDs here for special commands

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITING
# ============================================================================

user_last_request: Dict[int, float] = defaultdict(float)
user_request_count: Dict[int, int] = defaultdict(int)

def check_rate_limit(user_id: int) -> tuple[bool, int]:
    """Check if user is rate limited. Returns (is_allowed, seconds_remaining)"""
    current_time = time.time()
    last_request = user_last_request[user_id]
    
    if current_time - last_request < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - (current_time - last_request))
        return False, remaining
    
    return True, 0

def update_rate_limit(user_id: int):
    """Update rate limit tracking for user"""
    user_last_request[user_id] = time.time()
    user_request_count[user_id] += 1

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def split_message(text: str, max_length: int = 4000) -> list[str]:
    """Split long messages into chunks"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        
        # Try to split at newline
        split_pos = text.rfind('\n', 0, max_length)
        if split_pos == -1:
            split_pos = max_length
        
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()
    
    return chunks

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def escape_markdown(text: str) -> str:
    """Escape special characters for Markdown"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

# ============================================================================
# COMMAND HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user = update.effective_user
    logger.info(f"User {user.id} ({user.username}) started bot")
    
    welcome_text = f"""
🤖 *Intelligence Agent System*

Welcome {user.first_name}! I'm an autonomous AI agent designed for intelligence analysis and investigations.

*🎯 Commands:*
/start \\- Show this help message
/mission \\<description\\> \\- Execute intelligence mission
/status \\- Check system status
/stats \\- Your usage statistics
/help \\- Detailed help and examples

*💡 Quick Start:*
Just send me your investigation objective directly, or use:
`/mission Investigate LockBit ransomware activities`

*🔥 Features:*
• Autonomous multi\\-step planning
• Semantic document search
• Comprehensive report generation
• Real\\-time execution tracking

Ready to start? Send me your first mission\\!
"""
    
    keyboard = [
        [InlineKeyboardButton("📚 Examples", callback_data='examples')],
        [InlineKeyboardButton("ℹ️ About", callback_data='about')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=reply_markup
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = """
📖 *Detailed Help*

*Basic Usage:*
You can interact with me in two ways:

1️⃣ *Direct Message* \\(Recommended\\)
   Just type your mission naturally:
   `Analyze recent DDoS attacks on government sites`

2️⃣ *Command Format*
   Use the /mission command:
   `/mission Investigate phishing campaigns`

*📝 Mission Examples:*

🔍 *Threat Intelligence:*
   `Investigate APT29 activities in 2024`
   `Analyze LockBit ransomware TTPs`

🌐 *Cyber Attacks:*
   `Research recent DDoS attacks on financial sector`
   `Find information about SolarWinds supply chain attack`

🛡️ *Vulnerabilities:*
   `What are the latest critical CVEs in Apache?`
   `Analyze Log4Shell vulnerability impact`

*⚙️ Advanced Features:*

• *Multi\\-step Planning:* AI automatically breaks down complex investigations
• *Semantic Search:* Finds relevant information beyond keyword matching
• *Auto\\-summarization:* Condenses long documents for quick insights
• *Execution Logs:* See exactly what the agent did

*⏱️ Rate Limits:*
• 1 mission per minute per user
• No daily limit \\(fair use policy\\)

*❓ Need Help?*
Contact @YourSupport for assistance
"""
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check system status"""
    status_msg = await update.message.reply_text("🔄 Checking system status...")
    
    try:
        response = requests.get(API_HEALTH, timeout=10)
        data = response.json()
        
        if response.ok and data.get('status') == 'healthy':
            status_text = f"""
📊 *System Status*

✅ *Status:* {data['status'].upper()}
📚 *Documents:* {data['documents']:,}
💾 *Cache Size:* {data['cache_size'] / (1024*1024):.2f} MB
⏱️ *Uptime:* {format_duration(data['uptime'])}
🕐 *Last Check:* {datetime.now().strftime('%H:%M:%S')}

All systems operational ✅
"""
        else:
            status_text = "⚠️ *System Status*\n\nSystem is experiencing issues. Please try again later."
        
        await status_msg.edit_text(
            escape_markdown(status_text) if '\\' not in status_text else status_text,
            parse_mode=ParseMode.MARKDOWN_V2
        )
        
    except requests.exceptions.Timeout:
        await status_msg.edit_text("⏱️ Request timed out. System may be busy.")
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        await status_msg.edit_text(f"❌ Failed to check status: {str(e)}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show user statistics"""
    user_id = update.effective_user.id
    request_count = user_request_count.get(user_id, 0)
    
    stats_text = f"""
📈 *Your Statistics*

👤 *User ID:* `{user_id}`
📊 *Missions Executed:* {request_count}
⏰ *Member Since:* Session start

Keep investigating\\! 🔍
"""
    
    await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN_V2)

async def mission_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Execute intelligence mission"""
    user_id = update.effective_user.id
    user_name = update.effective_user.username or update.effective_user.first_name
    
    # Check rate limit
    allowed, remaining = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(
            f"⏳ Please wait {remaining}s before submitting another mission.\n\n"
            "This helps ensure fair usage for all users."
        )
        return
    
    # Get mission text
    if not context.args:
        await update.message.reply_text(
            "❌ Please provide a mission description.\n\n"
            "Example: `/mission Investigate ransomware attacks`\n"
            "Or just send your mission directly without the command!",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    mission_text = ' '.join(context.args)
    
    if len(mission_text) < 10:
        await update.message.reply_text(
            "❌ Mission description too short. Please provide at least 10 characters."
        )
        return
    
    # Update rate limit
    update_rate_limit(user_id)
    
    # Log mission
    logger.info(f"Mission from {user_name} ({user_id}): {mission_text[:100]}...")
    
    # Create status message with inline keyboard
    keyboard = [[InlineKeyboardButton("❌ Cancel", callback_data=f'cancel_{user_id}')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    status_message = await update.message.reply_text(
        f"🔄 *Executing Mission*\n\n"
        f"_{mission_text}_\n\n"
        f"⏳ AI agent is planning and executing...",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )
    
    try:
        # Call API
        start_time = time.time()
        response = requests.post(
            API_EXECUTE,
            json={"mission": mission_text},
            timeout=300  # 5 minutes
        )
        duration = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            # Format report
            report = result["final_report"]
            metadata = result.get("metadata", {})
            
            # Add metadata footer
            footer = f"\n\n─────────────────\n"
            footer += f"⏱️ Completed in {metadata.get('total_duration_seconds', duration):.1f}s\n"
            footer += f"✅ {metadata.get('successful_steps', 0)}/{metadata.get('total_steps', 0)} steps successful"
            
            full_message = f"✅ *Mission Complete*\n\n{report}{footer}"
            
            # Split and send if too long
            chunks = split_message(full_message, max_length=4000)
            
            await status_message.delete()
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    # First chunk with execution log button
                    keyboard = [[InlineKeyboardButton("📋 View Execution Log", callback_data=f'log_{user_id}')]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await update.message.reply_text(
                        chunk,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=reply_markup
                    )
                    
                    # Store execution log in context
                    context.user_data[f'exec_log_{user_id}'] = result.get("execution_log", [])
                else:
                    await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
                    await asyncio.sleep(0.5)  # Prevent rate limiting
        
        else:
            error_msg = result.get("error", "Unknown error")
            await status_message.edit_text(
                f"❌ *Mission Failed*\n\n"
                f"Error: {error_msg}\n\n"
                f"Please try again or refine your mission description.",
                parse_mode=ParseMode.MARKDOWN
            )
    
    except requests.exceptions.Timeout:
        await status_message.edit_text(
            "⏱️ *Request Timeout*\n\n"
            "The mission is taking longer than expected. "
            "It may still be processing. Please try checking status in a moment.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    except requests.exceptions.HTTPError as e:
        await status_message.edit_text(
            f"❌ *HTTP Error*\n\n"
            f"Status Code: {e.response.status_code}\n"
            f"Please try again later.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    except Exception as e:
        logger.error(f"Mission execution error: {e}", exc_info=True)
        await status_message.edit_text(
            f"❌ *Error*\n\n"
            f"An unexpected error occurred: {str(e)[:200]}\n\n"
            f"Please contact support if this persists.",
            parse_mode=ParseMode.MARKDOWN
        )

async def handle_direct_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle direct messages as mission commands"""
    message_text = update.message.text.strip()
    
    # Treat message as mission
    context.args = message_text.split()
    await mission_command(update, context)

# ============================================================================
# CALLBACK HANDLERS
# ============================================================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    
    if data == 'examples':
        examples_text = """
📚 *Mission Examples*

*🔍 Threat Intelligence:*
• `Investigate APT29 Cozy Bear activities`
• `Analyze LockBit 3.0 ransomware TTPs`
• `Research Lazarus Group recent campaigns`

*🌐 Cyber Attacks:*
• `DDoS attacks on financial institutions 2024`
• `Phishing campaigns targeting healthcare`
• `Recent supply chain attacks analysis`

*🛡️ Vulnerabilities:*
• `Critical CVEs in Apache products`
• `Log4Shell vulnerability exploitation`
• `Zero-day vulnerabilities in Microsoft Exchange`

*📊 Trends:*
• `Emerging ransomware trends in 2024`
• `AI-powered cyber attacks analysis`
• `Cryptocurrency theft techniques`

Just copy and send any example\\!
"""
        await query.edit_message_text(
            examples_text,
            parse_mode=ParseMode.MARKDOWN_V2
        )
    
    elif data == 'about':
        about_text = """
ℹ️ *About Intelligence Agent*

This bot is powered by an advanced AI system that:

🧠 *Autonomous Planning*
   Uses AI to break down complex investigations

🔍 *Semantic Search*
   Finds relevant info beyond keywords

📄 *Document Analysis*
   Retrieves and analyzes full documents

📊 *Report Generation*
   Creates comprehensive intelligence reports

*🔧 Technology:*
• Cerebras AI for reasoning
• FAISS for vector search
• Sentence Transformers for embeddings
• Modal for scalable deployment

*📖 Version:* 2\\.0\\.0
*🏢 Platform:* Intelligence Agent System
"""
        await query.edit_message_text(
            about_text,
            parse_mode=ParseMode.MARKDOWN_V2
        )
    
    elif data.startswith('log_'):
        # Show execution log
        log_user_id = int(data.split('_')[1])
        
        if log_user_id != user_id:
            await query.answer("❌ This is not your mission!", show_alert=True)
            return
        
        exec_log = context.user_data.get(f'exec_log_{user_id}', [])
        
        if not exec_log:
            await query.answer("No execution log available", show_alert=True)
            return
        
        # Format execution log
        log_text = "📋 *Execution Log*\n\n"
        for step in exec_log:
            status_icon = "✅" if step.get('result', {}).get('status') == 'success' else "❌"
            log_text += f"{status_icon} *Step {step['step']}:* {step['tool']}\n"
            log_text += f"   _{step.get('description', 'No description')}_\n"
            log_text += f"   Duration: {step.get('duration_seconds', 0):.2f}s\n\n"
        
        # Send as new message
        await query.message.reply_text(log_text, parse_mode=ParseMode.MARKDOWN)
        await query.answer("Execution log sent!")
    
    elif data.startswith('cancel_'):
        cancel_user_id = int(data.split('_')[1])
        
        if cancel_user_id != user_id:
            await query.answer("❌ This is not your mission!", show_alert=True)
            return
        
        await query.edit_message_text(
            "❌ *Mission Cancelled*\n\n"
            "The mission has been cancelled by user request.",
            parse_mode=ParseMode.MARKDOWN
        )
        await query.answer("Mission cancelled")

# ============================================================================
# ADMIN COMMANDS (Optional)
# ============================================================================

async def admin_add_doc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to add documents (requires admin rights)"""
    user_id = update.effective_user.id
    
    if user_id not in ADMIN_USER_IDS and ADMIN_USER_IDS:
        await update.message.reply_text("❌ You don't have permission to use this command.")
        return
    
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: `/add_doc <text> [url] [title]`\n\n"
            "Example: `/add_doc \"Important security document content\" https://example.com \"Security Report\"`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    text = context.args[0]
    url = context.args[1] if len(context.args) > 1 else ""
    title = context.args[2] if len(context.args) > 2 else ""
    
    try:
        response = requests.post(
            API_ADD_DOC,
            params={"text": text, "url": url, "title": title},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get("status") == "success":
            await update.message.reply_text(
                f"✅ Document added successfully!\n"
                f"Document ID: {result.get('doc_id')}",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                f"❌ Failed to add document: {result.get('detail', 'Unknown error')}"
            )
    
    except Exception as e:
        logger.error(f"Add document failed: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin command to view global statistics"""
    user_id = update.effective_user.id
    
    if user_id not in ADMIN_USER_IDS and ADMIN_USER_IDS:
        await update.message.reply_text("❌ Admin only command.")
        return
    
    total_users = len(user_request_count)
    total_requests = sum(user_request_count.values())
    
    stats = f"""
🔐 *Admin Statistics*

👥 *Total Users:* {total_users}
📊 *Total Missions:* {total_requests}
⏰ *Bot Uptime:* {format_duration(time.time() - bot_start_time)}

*Top Users:*
"""
    
    # Get top 5 users
    top_users = sorted(user_request_count.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (uid, count) in enumerate(top_users, 1):
        stats += f"{i}. User {uid}: {count} missions\n"
    
    await update.message.reply_text(stats, parse_mode=ParseMode.MARKDOWN)

# ============================================================================
# ERROR HANDLER
# ============================================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Exception while handling update: {context.error}", exc_info=context.error)
    
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "❌ An error occurred while processing your request. "
            "Please try again or contact support if the issue persists."
        )

# ============================================================================
# MAIN FUNCTION
# ============================================================================

bot_start_time = time.time()

def main():
    """Start the bot"""
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ Error: Please set TELEGRAM_BOT_TOKEN environment variable or update the code")
        return
    
    logger.info("🤖 Starting Intelligence Agent Telegram Bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("mission", mission_command))
    
    # Admin commands
    if ADMIN_USER_IDS:
        application.add_handler(CommandHandler("add_doc", admin_add_doc))
        application.add_handler(CommandHandler("admin_stats", admin_stats))
    
    # Callback handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Message handler for direct messages
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_direct_message
    ))
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start bot
    logger.info("✅ Bot started successfully!")
    logger.info(f"📡 Connected to API: {API_BASE_URL}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
