"""Telegram bot frontend for multimodal AI agent."""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from PIL import Image
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..agent import MultimodalAgent, create_agent
from ..config import settings
from ..models import create_multimodal_message


class TelegramBot:
    """
    Telegram bot interface for the multimodal AI agent.

    Handles:
    - Text messages
    - Images (photos)
    - Audio files
    - Video files
    - Voice messages
    - Multi-media messages (e.g., photo with caption)
    """

    def __init__(self, token: str):
        """
        Initialize Telegram bot.

        Args:
            token: Telegram bot token
        """
        self.token = token
        self.agent: Optional[MultimodalAgent] = None
        self.application: Optional[Application] = None

        # Store temporary files per user conversation
        self.temp_files: Dict[str, list] = {}

    async def initialize_agent(self):
        """Initialize the multimodal agent."""
        logger.info("Initializing multimodal agent...")
        self.agent = await create_agent(use_mcp=True)
        logger.info("✓ Agent initialized")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        logger.info(f"User {user.id} started conversation")

        welcome_message = (
            f"👋 Hello {user.first_name}!\n\n"
            "I'm a multimodal AI assistant powered by Qwen3 Omni. "
            "I can help you with:\n\n"
            "📝 Text conversations\n"
            "🖼️ Image analysis\n"
            "🎵 Audio processing\n"
            "🎬 Video understanding\n"
            "🔧 Tool usage via MCP servers\n\n"
            "Just send me text, images, audio, or video and I'll help you out!"
        )

        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = (
            "🤖 **Multimodal AI Assistant Help**\n\n"
            "**Commands:**\n"
            "/start - Start conversation\n"
            "/help - Show this help message\n"
            "/clear - Clear conversation history\n\n"
            "**Supported Inputs:**\n"
            "• Text messages\n"
            "• Photos/Images\n"
            "• Audio files\n"
            "• Video files\n"
            "• Voice messages\n"
            "• Documents\n\n"
            "**Multi-modal Messages:**\n"
            "You can send images with captions, and I'll analyze both!\n\n"
            "**Examples:**\n"
            "• Send a photo: \"What's in this image?\"\n"
            "• Send audio: \"Transcribe this audio\"\n"
            "• Send video: \"Summarize this video\"\n"
            "• Send text: \"Explain quantum computing\""
        )

        await update.message.reply_text(help_message, parse_mode="Markdown")

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command."""
        user_id = str(update.effective_user.id)

        # Clear temporary files
        if user_id in self.temp_files:
            for file_path in self.temp_files[user_id]:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file_path}: {e}")

            self.temp_files[user_id] = []

        await update.message.reply_text("✓ Conversation history cleared!")

    async def _download_file(self, file_id: str, context: ContextTypes.DEFAULT_TYPE) -> str:
        """
        Download a file from Telegram.

        Args:
            file_id: Telegram file ID
            context: Bot context

        Returns:
            Path to downloaded file
        """
        file = await context.bot.get_file(file_id)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.file_path).suffix)
        await file.download_to_drive(temp_file.name)
        return temp_file.name

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle incoming messages (text, images, audio, video).

        Args:
            update: Telegram update
            context: Bot context
        """
        if not self.agent:
            await self.initialize_agent()

        user_id = str(update.effective_user.id)
        message = update.message

        # Initialize temp files list for user
        if user_id not in self.temp_files:
            self.temp_files[user_id] = []

        try:
            # Send typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

            # Extract message content
            text = message.text or message.caption or ""
            images = []
            audio = None
            video = None

            # Handle photos
            if message.photo:
                logger.info(f"Received photo from user {user_id}")
                # Get highest resolution photo
                photo = message.photo[-1]
                photo_file = await context.bot.get_file(photo.file_id)

                # Download to memory
                photo_bytes = io.BytesIO()
                await photo_file.download_to_memory(photo_bytes)
                photo_bytes.seek(0)

                # Load as PIL Image
                image = Image.open(photo_bytes)
                images.append(image)

            # Handle audio
            if message.audio or message.voice:
                logger.info(f"Received audio from user {user_id}")
                audio_file_id = message.audio.file_id if message.audio else message.voice.file_id
                audio = await self._download_file(audio_file_id, context)
                self.temp_files[user_id].append(audio)

            # Handle video
            if message.video or message.video_note:
                logger.info(f"Received video from user {user_id}")
                video_file_id = message.video.file_id if message.video else message.video_note.file_id
                video = await self._download_file(video_file_id, context)
                self.temp_files[user_id].append(video)

            # Handle documents (could be images, audio, video)
            if message.document:
                logger.info(f"Received document from user {user_id}")
                mime_type = message.document.mime_type or ""

                if mime_type.startswith("image/"):
                    doc_file = await context.bot.get_file(message.document.file_id)
                    doc_bytes = io.BytesIO()
                    await doc_file.download_to_memory(doc_bytes)
                    doc_bytes.seek(0)
                    image = Image.open(doc_bytes)
                    images.append(image)
                elif mime_type.startswith("audio/"):
                    audio = await self._download_file(message.document.file_id, context)
                    self.temp_files[user_id].append(audio)
                elif mime_type.startswith("video/"):
                    video = await self._download_file(message.document.file_id, context)
                    self.temp_files[user_id].append(video)

            # Create multimodal message
            if not text and not images and not audio and not video:
                await message.reply_text("Please send text, image, audio, or video.")
                return

            user_msg = create_multimodal_message(
                text=text or "Please analyze this media.",
                images=images if images else None,
                audio=audio,
                video=video,
            )

            logger.info(f"Processing message from user {user_id}")

            # Get agent response
            response = await self.agent.ainvoke(user_msg, user_id=user_id)

            # Send response
            response_text = response.content
            if isinstance(response_text, list):
                # Extract text from content list
                response_text = "\n".join(
                    item.get("text", "") for item in response_text if item.get("type") == "text"
                )

            await message.reply_text(response_text or "I processed your message.")

            logger.info(f"Response sent to user {user_id}")

        except Exception as e:
            logger.error(f"Error processing message from user {user_id}: {e}", exc_info=True)
            await message.reply_text(
                "Sorry, I encountered an error processing your message. Please try again."
            )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        logger.error(f"Update {update} caused error: {context.error}", exc_info=context.error)

    def build_application(self) -> Application:
        """
        Build and configure the Telegram application.

        Returns:
            Configured Application instance
        """
        # Create application
        self.application = Application.builder().token(self.token).build()

        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))

        # Add message handlers
        self.application.add_handler(
            MessageHandler(
                filters.TEXT | filters.PHOTO | filters.AUDIO | filters.VOICE |
                filters.VIDEO | filters.Document.ALL,
                self.handle_message,
            )
        )

        # Add error handler
        self.application.add_error_handler(self.error_handler)

        return self.application

    async def start(self):
        """Start the Telegram bot."""
        logger.info("Starting Telegram bot...")

        # Initialize agent
        await self.initialize_agent()

        # Build application
        self.build_application()

        # Start polling
        logger.info("✓ Bot started and polling for messages")
        await self.application.run_polling(allowed_updates=Update.ALL_TYPES)

    async def stop(self):
        """Stop the Telegram bot."""
        logger.info("Stopping Telegram bot...")

        if self.application:
            await self.application.stop()

        # Clean up temp files
        for user_files in self.temp_files.values():
            for file_path in user_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file_path}: {e}")

        logger.info("✓ Bot stopped")


async def run_bot():
    """Run the Telegram bot."""
    bot = TelegramBot(token=settings.telegram_bot_token)
    await bot.start()


if __name__ == "__main__":
    # Configure logging
    logger.add(
        "logs/telegram_bot.log",
        rotation="1 day",
        retention="7 days",
        level=settings.log_level,
    )

    # Run bot
    asyncio.run(run_bot())
