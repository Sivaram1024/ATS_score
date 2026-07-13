import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration, pulled from environment variables."""

    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-key-change-me")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload cap
    ALLOWED_EXTENSIONS = {"pdf"}

    # How many chat turns to keep per case file before trimming context
    MAX_CHAT_HISTORY = 12

    @staticmethod
    def validate():
        missing = []
        if not Config.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        return missing
