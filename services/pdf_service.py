"""Handles resume PDF ingestion: validation + text extraction."""

import logging
from io import BytesIO

from pdfminer.high_level import extract_text

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Raised when a resume PDF can't be read or is unusable."""


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def extract_resume_text(file_storage) -> str:
    """
    Extract text from an uploaded PDF (Werkzeug FileStorage).
    Raises PDFExtractionError with a user-facing message on failure.
    """
    try:
        raw = file_storage.read()
        if not raw:
            raise PDFExtractionError("The uploaded file is empty.")

        text = extract_text(BytesIO(raw))
        text = text.strip() if text else ""

        if len(text) < 50:
            raise PDFExtractionError(
                "Couldn't find readable text in this PDF. "
                "If it's a scanned image, try a text-based export instead."
            )
        return text

    except PDFExtractionError:
        raise
    except Exception as exc:
        logger.exception("PDF extraction failed")
        raise PDFExtractionError(f"Couldn't read this PDF: {exc}") from exc
