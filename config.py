"""
Application configuration and environment variable loading.
"""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration parameters for ATS Score & Telegram Copilot."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    BERT_MODEL_NAME: str = os.getenv("BERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    MAX_CONTENT_LENGTH: int = 8 * 1024 * 1024  # 8 MB file size cap
    ALLOWED_EXTENSIONS: set = {"pdf"}

    MAX_CHAT_HISTORY: int = 12

    @staticmethod
    def validate() -> List[str]:
        """Validates critical credentials, returning list of missing keys."""
        missing = []
        if not Config.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if not Config.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        return missing
