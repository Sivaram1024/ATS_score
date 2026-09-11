"""
Retrieval-Augmented Generation utilities supporting multi-resume chunking on CPU.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

import re
import numpy as np

from services.similarity_service import get_model

CHUNK_MAX_WORDS = 60


def _split_into_chunks(text: str, source: str) -> list:
    raw_blocks = re.split(r"\n\s*\n|\n(?=[•\-\*•])", text)
    chunks = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        words = block.split()
        for i in range(0, len(words), CHUNK_MAX_WORDS):
            piece = " ".join(words[i:i + CHUNK_MAX_WORDS]).strip()
            if piece:
                chunks.append({"source": source, "text": piece})
    return chunks


def build_and_embed_chunks(resume_text: str, job_desc: str) -> list:
    """Legacy single-resume chunk builder."""
    return build_multi_resume_chunks([{"filename": "Resume", "text": resume_text}], job_desc)


def build_multi_resume_chunks(resumes: list[dict], job_desc: str) -> list:
    """Indexes chunks for 1 or more resumes and the job description."""
    chunks = []
    for idx, r in enumerate(resumes, start=1):
        fn = r.get("filename", f"Candidate_{idx}")
        chunks += _split_into_chunks(r.get("text", ""), f"RESUME: {fn}")

    chunks += _split_into_chunks(job_desc, "JOB DESCRIPTION")
    if not chunks:
        return []

    model = get_model()
    vectors = model.encode([c["text"] for c in chunks], device="cpu")
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def _cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve(query: str, chunks: list, top_k: int = 6) -> list:
    if not chunks or not query.strip():
        return []

    model = get_model()
    query_vec = model.encode([query], device="cpu")[0]

    scored = [(_cosine(query_vec, c["embedding"]), c) for c in chunks if "embedding" in c]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def format_context(chunks: list) -> str:
    if not chunks:
        return "(no relevant passages retrieved)"
    return "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
