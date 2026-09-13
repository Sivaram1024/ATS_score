"""
Semantic text similarity evaluation using Sentence-BERT embeddings strictly on CPU.
Guarantees deterministic execution across ZeroGPU, Hugging Face Spaces, and local runners.
"""

import logging
import os
import threading

# Force CPU mode globally to avoid CUDA memory contention on cloud/ZeroGPU environments
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config

logger = logging.getLogger(__name__)

_model_instance = None
_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """
    Thread-safe lazy initialization of Sentence-BERT model on CPU.
    """
    global _model_instance
    if _model_instance is None:
        with _lock:
            if _model_instance is None:
                logger.info("Initializing Sentence-BERT on CPU: %s", Config.BERT_MODEL_NAME)
                _model_instance = SentenceTransformer(Config.BERT_MODEL_NAME, device="cpu")
    return _model_instance


def calculate_similarity(resume_text: str, job_desc: str) -> float:
    """
    Computes normalized cosine similarity in range [0.0, 1.0] between resume text and job description.
    Falls back gracefully to 0.5 if evaluation fails.
    """
    if not resume_text.strip() or not job_desc.strip():
        return 0.0

    try:
        model = get_model()
        embeddings = model.encode([resume_text, job_desc], device="cpu", show_progress_bar=False)
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, min(1.0, float(sim)))
    except Exception as err:
        logger.warning("Similarity calculation fallback: %s", err)
        return 0.5
