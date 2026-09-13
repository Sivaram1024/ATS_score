"""
Comprehensive unit tests for ATS Score & Telegram Copilot modules.
Validates:
- PDF document extraction & error boundaries
- CaseStore multi-resume session queueing & caching
- Sentence-BERT CPU similarity evaluation
- RAG segmentation and semantic vector retrieval
- Defensive blended ATS scoring rubric and mismatch penalties
- Configuration validation
"""

import os
import io
import pytest

from config import Config
from services.case_store import (
    create_case,
    get_case,
    update_case,
    add_resume,
    get_resumes,
    clear_pending_resumes,
    append_chat,
    delete_case,
)
from services.pdf_service import allowed_file, extract_resume_text, PDFExtractionError
from services.similarity_service import calculate_similarity, get_model
from services.rag_service import (
    _segment_text,
    build_and_embed_chunks,
    build_multi_resume_chunks,
    retrieve,
    format_context,
)
from services.gemini_service import compute_blended_ats_score


# ---------------------------------------------------------------------------
# 1. Config Tests
# ---------------------------------------------------------------------------
def test_config_defaults():
    assert Config.MAX_CONTENT_LENGTH == 8 * 1024 * 1024
    assert "pdf" in Config.ALLOWED_EXTENSIONS
    assert Config.MAX_CHAT_HISTORY >= 5
    assert isinstance(Config.validate(), list)


# ---------------------------------------------------------------------------
# 2. PDF Service Tests
# ---------------------------------------------------------------------------
def test_pdf_allowed_file():
    assert allowed_file("resume.pdf", {"pdf"}) is True
    assert allowed_file("resume.PDF", {"pdf"}) is True
    assert allowed_file("script.py", {"pdf"}) is False
    assert allowed_file("no_extension", {"pdf"}) is False


def test_pdf_empty_stream_raises():
    empty_stream = io.BytesIO(b"")
    with pytest.raises(PDFExtractionError, match="empty"):
        extract_resume_text(empty_stream)


def test_pdf_short_text_raises():
    # Synthetic mock of short/unreadable text
    short_stream = io.BytesIO(b"%PDF-1.4 short dummy")
    with pytest.raises(PDFExtractionError):
        extract_resume_text(short_stream)


# ---------------------------------------------------------------------------
# 3. Case Store (Session Lifecycle) Tests
# ---------------------------------------------------------------------------
def test_case_store_lifecycle():
    cid = create_case()
    case = get_case(cid)
    assert case is not None
    assert case["resumes"] == []
    assert case["chat_history"] == []

    # Add single resume
    c1 = add_resume(cid, "cand1.pdf", "Senior Python Developer with 5 years experience in machine learning.")
    assert c1 == 1

    # Add second resume
    c2 = add_resume(cid, "cand2.pdf", "Frontend React developer with TypeScript and CSS.")
    assert c2 == 2

    # Deduplication check: updating cand1.pdf updates text rather than appending
    c3 = add_resume(cid, "cand1.pdf", "Updated Senior Python Developer text with PyTorch.")
    assert c3 == 2

    resumes = get_resumes(cid)
    assert len(resumes) == 2
    assert "PyTorch" in resumes[0]["text"]

    # Append chat
    append_chat(cid, "user", "Who is better?")
    append_chat(cid, "model", "Candidate 1 is stronger in Python.")
    assert len(get_case(cid)["chat_history"]) == 2

    # Clear pending resumes
    clear_pending_resumes(cid)
    assert len(get_resumes(cid)) == 0

    # Delete case
    delete_case(cid)
    assert get_case(cid) is None


# ---------------------------------------------------------------------------
# 4. Sentence-BERT CPU Similarity Tests
# ---------------------------------------------------------------------------
def test_bert_cpu_device():
    model = get_model()
    assert model.device.type == "cpu"


def test_similarity_calculation():
    text1 = "Expert Python backend engineer specializing in FastAPI, PostgreSQL, and Docker."
    text2 = "Senior Python developer with experience building microservices and relational databases."
    text3 = "Pastry chef with experience in French baking and wedding cakes."

    sim_high = calculate_similarity(text1, text2)
    sim_low = calculate_similarity(text1, text3)

    assert 0.0 <= sim_high <= 1.0
    assert 0.0 <= sim_low <= 1.0
    assert sim_high > sim_low

    # Empty inputs
    assert calculate_similarity("", "some text") == 0.0


# ---------------------------------------------------------------------------
# 5. RAG Service Tests
# ---------------------------------------------------------------------------
def test_rag_segmentation_and_retrieval():
    doc1 = """Python Developer Experience:
    - Built RESTful APIs using FastAPI and Pydantic.
    - Implemented database migrations with Alembic and SQLAlchemy.
    - Containerized microservices with Docker and orchestrated with Kubernetes.
    """
    doc2 = "Job Description: Looking for a FastAPI engineer with strong Docker skills."

    resumes = [{"filename": "Alice.pdf", "text": doc1}]
    passages = build_multi_resume_chunks(resumes, doc2)
    assert len(passages) > 0

    retrieved = retrieve("FastAPI and Docker", passages, top_k=2)
    assert len(retrieved) <= 2
    formatted = format_context(retrieved)
    assert "FastAPI" in formatted or "Docker" in formatted


# ---------------------------------------------------------------------------
# 6. Blended Scoring & Penalties Tests
# ---------------------------------------------------------------------------
def test_blended_ats_score_strong_match():
    report = {
        "role_alignment": "Strong Match",
        "calculated_score": 92,
        "matched_skills": ["Python", "FastAPI", "Docker", "PostgreSQL", "Git"],
        "missing_skills": ["AWS"],
        "partial_matches": ["CI/CD"],
    }
    score, rating = compute_blended_ats_score(report, bert_similarity=0.85)
    assert score >= 80.0
    assert rating in ["Strong Match", "Good Match"]


def test_blended_ats_score_unrelated_domain_penalty():
    report = {
        "role_alignment": "Unrelated Domain",
        "calculated_score": 15,
        "matched_skills": [],
        "missing_skills": ["Python", "Machine Learning", "FastAPI", "SQL"],
        "partial_matches": [],
    }
    score, rating = compute_blended_ats_score(report, bert_similarity=0.35)
    assert score <= 20.0
    assert rating == "Unrelated Role"


def test_blended_ats_score_missing_core_skills_penalty():
    report = {
        "role_alignment": "Moderate Match",
        "calculated_score": 70,
        "matched_skills": [],
        "missing_skills": ["Python", "SQL", "Machine Learning"],
        "partial_matches": [],
    }
    score, rating = compute_blended_ats_score(report, bert_similarity=0.50)
    # Zero matched skills caps score at 25
    assert score <= 25.0
