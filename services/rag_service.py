"""
Retrieval-Augmented Generation (RAG) vector index and semantic search utilities.
Performs document segmentation, embedding generation, and cosine similarity retrieval on CPU.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import re
from typing import Any, Dict, List
import numpy as np

from services.similarity_service import get_model

CHUNK_MAX_WORDS = 60


def _segment_text(content: str, source_label: str) -> List[Dict[str, Any]]:
    """Segments document into semantic passages respecting paragraph breaks and bullet points."""
    paragraphs = re.split(r"\n\s*\n|\n(?=[\u2022\-\*●])", content)
    passages = []
    for para in paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        words = cleaned.split()
        for idx in range(0, len(words), CHUNK_MAX_WORDS):
            sub = " ".join(words[idx:idx + CHUNK_MAX_WORDS]).strip()
            if sub:
                passages.append({"source": source_label, "text": sub})
    return passages


def build_and_embed_chunks(resume_text: str, job_desc: str) -> List[Dict[str, Any]]:
    """Builds vector index for single-resume evaluation."""
    return build_multi_resume_chunks([{"filename": "Candidate", "text": resume_text}], job_desc)


def build_multi_resume_chunks(resumes: List[Dict[str, Any]], job_desc: str) -> List[Dict[str, Any]]:
    """
    Indexes and embeds semantic passages for multiple candidate resumes and target job description.
    """
    passages: List[Dict[str, Any]] = []
    for index, item in enumerate(resumes, start=1):
        name = item.get("filename", f"Candidate_{index}")
        passages.extend(_segment_text(item.get("text", ""), f"RESUME: {name}"))

    passages.extend(_segment_text(job_desc, "JOB DESCRIPTION"))

    if not passages:
        return []

    model = get_model()
    vectors = model.encode([p["text"] for p in passages], device="cpu", show_progress_bar=False)
    for passage, vec in zip(passages, vectors):
        passage["embedding"] = vec
    return passages


def _cosine_dist(v1: np.ndarray, v2: np.ndarray) -> float:
    a, b = np.asarray(v1), np.asarray(v2)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm_product) if norm_product > 0 else 0.0


def retrieve(query: str, passages: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    """Retrieves top-K most semantically relevant document passages for the query."""
    if not passages or not query.strip():
        return []

    model = get_model()
    q_vec = model.encode([query], device="cpu", show_progress_bar=False)[0]

    ranked = [(_cosine_dist(q_vec, p["embedding"]), p) for p in passages if "embedding" in p]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:top_k]]


def format_context(passages: List[Dict[str, Any]]) -> str:
    """Formats retrieved passages into clean prompt context."""
    if not passages:
        return "(no relevant document passages found)"
    return "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)
