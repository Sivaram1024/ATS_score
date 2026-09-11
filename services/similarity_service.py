"""Semantic similarity between resume and job description using Sentence-BERT on CPU."""

import logging
import os
import threading

# Force CPU mode globally to avoid any CUDA initialization on cloud/ZeroGPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Lazily load the model once per process explicitly on CPU (thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading BERT model on CPU: %s", Config.BERT_MODEL_NAME)
                _model = SentenceTransformer(Config.BERT_MODEL_NAME, device="cpu")
    return _model


def calculate_similarity(resume_text: str, job_desc: str) -> float:
    """Returns cosine similarity in [0, 1] between resume and job description strictly on CPU."""
    try:
        model = get_model()
        embeddings = model.encode([resume_text, job_desc], device="cpu")
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, min(1.0, float(score)))
    except Exception as e:
        logger.warning(f"Similarity calculation fallback: {e}")
        return 0.5
