"""Semantic similarity between resume and job description using Sentence-BERT."""

import logging
import threading

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Lazily load the model once per process (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading BERT model: %s", Config.BERT_MODEL_NAME)
                _model = SentenceTransformer(Config.BERT_MODEL_NAME)
    return _model


def calculate_similarity(resume_text: str, job_desc: str) -> float:
    """Returns cosine similarity in [0, 1] between resume and job description."""
    model = get_model()
    embeddings = model.encode([resume_text, job_desc])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    # Clamp for safety — cosine sim can drift a hair outside [0,1] on float rounding
    return max(0.0, min(1.0, float(score)))
