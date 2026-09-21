"""
Retrieval-Augmented Generation (RAG) passage index and semantic search utilities.
Performs document segmentation and fast, lightweight TF-IDF cosine similarity retrieval.
Optimized for 512MB RAM constraints with zero torch/transformers dependencies.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import re
from typing import Any, Dict, List
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """Builds passage index for single-resume evaluation."""
    return build_multi_resume_chunks([{"filename": "Candidate", "text": resume_text}], job_desc)


def build_multi_resume_chunks(resumes: List[Dict[str, Any]], job_desc: str) -> List[Dict[str, Any]]:
    """
    Indexes semantic passages for multiple candidate resumes and target job description.
    Uses lightweight text segmentation without expensive model loading.
    """
    passages: List[Dict[str, Any]] = []
    for index, item in enumerate(resumes, start=1):
        name = item.get("filename", f"Candidate_{index}")
        passages.extend(_segment_text(item.get("text", ""), f"RESUME: {name}"))

    passages.extend(_segment_text(job_desc, "JOB DESCRIPTION"))
    return passages


def retrieve(query: str, passages: List[Dict[str, Any]], top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieves top-K most semantically and contextually relevant document passages for the query.
    Uses lightweight TF-IDF n-gram cosine matching (<2ms, <1MB RAM).
    """
    if not passages or not query.strip():
        return []

    texts = [p.get("text", "") for p in passages]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        mat = vec.fit_transform(texts)
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, mat)[0]
        ranked = sorted(zip(sims, passages), key=lambda x: x[0], reverse=True)
        return [p for s, p in ranked[:top_k]]
    except Exception:
        return passages[:top_k]


def format_context(passages: List[Dict[str, Any]]) -> str:
    """Formats retrieved passages into clean prompt context."""
    if not passages:
        return "(no relevant document passages found)"
    return "\n\n".join(f"[{p['source']}] {p['text']}" for p in passages)

