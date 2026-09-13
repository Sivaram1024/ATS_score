"""
PDF document text extraction service for ATS resume evaluation.
Provides robust text extraction, format validation, and scanned-PDF detection.
"""

import logging
from io import BytesIO
from typing import Any

from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when resume PDF extraction fails or document content is unreadable."""
    pass


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Validates that the file has an approved extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def extract_resume_text(source: Any) -> str:
    """
    Extract plain text from an uploaded PDF file.
    Supports file-like objects (BytesIO, FileStorage), file paths, or raw bytes.
    Raises PDFExtractionError if the document is empty or non-extractable.
    """
    try:
        if isinstance(source, bytes):
            stream = BytesIO(source)
        elif hasattr(source, "read"):
            content = source.read()
            if not content:
                raise PDFExtractionError("The uploaded PDF file is empty.")
            stream = BytesIO(content) if isinstance(content, bytes) else BytesIO(content.encode("utf-8", errors="ignore"))
        elif isinstance(source, str):
            with open(source, "rb") as f:
                content = f.read()
            if not content:
                raise PDFExtractionError("The specified PDF file is empty.")
            stream = BytesIO(content)
        else:
            raise PDFExtractionError(f"Unsupported PDF input type: {type(source).__name__}")

        raw_text = extract_text(stream)
        text = raw_text.strip() if raw_text else ""

        if len(text) < 50:
            raise PDFExtractionError(
                "Could not extract sufficient text from this PDF. "
                "If the resume is a scanned image or flattened graphic, "
                "please provide a text-searchable PDF export."
            )
        return text

    except PDFExtractionError:
        raise
    except Exception as exc:
        logger.exception("PDF extraction error: %s", exc)
        raise PDFExtractionError(f"Unable to process this PDF: {exc}") from exc
