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
    set_job_desc,
    get_job_desc,
    set_state,
    get_state,
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


# ---------------------------------------------------------------------------
# 7. Workflow Session & Multi-Resume Queue Tests
# ---------------------------------------------------------------------------
def test_workflow_session_multi_resume_state():
    cid = create_case()
    assert get_state(cid) == "AWAITING_JD"
    assert get_job_desc(cid) == ""

    # 1. Store JD
    sample_jd = "Looking for a Python Developer with Docker and FastAPI experience."
    set_job_desc(cid, sample_jd)
    assert get_job_desc(cid) == sample_jd
    assert get_state(cid) == "AWAITING_RESUMES"

    # 2. Upload multiple resumes
    c1 = add_resume(cid, "Candidate_A.pdf", "Python developer with Docker and FastAPI experience.")
    assert c1 == 1
    assert get_job_desc(cid) == sample_jd  # JD must NOT be cleared or overwritten

    c2 = add_resume(cid, "Candidate_B.pdf", "Python backend engineer with SQL experience.")
    assert c2 == 2
    assert get_job_desc(cid) == sample_jd

    c3 = add_resume(cid, "Candidate_C.pdf", "Frontend developer with JavaScript.")
    assert c3 == 3
    assert get_job_desc(cid) == sample_jd

    resumes = get_resumes(cid)
    assert len(resumes) == 3
    assert [r["filename"] for r in resumes] == ["Candidate_A.pdf", "Candidate_B.pdf", "Candidate_C.pdf"]

    # 3. Simulate analysis transition
    set_state(cid, "ANALYZED")
    assert get_state(cid) == "ANALYZED"
    delete_case(cid)


# ---------------------------------------------------------------------------
# 8. HealthCheck HTTP Server Tests
# ---------------------------------------------------------------------------
def test_healthcheck_handler():
    from bot import HealthCheckHandler
    from unittest.mock import MagicMock

    handler = HealthCheckHandler.__new__(HealthCheckHandler)
    handler.path = "/health"
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    # Test GET /health
    handler.do_GET()
    handler.send_response.assert_called_with(200)
    body = handler.wfile.getvalue()
    assert b"status" in body and b"ok" in body

    # Test HEAD /health
    handler.send_response.reset_mock()
    handler.do_HEAD()
    handler.send_response.assert_called_with(200)

    # Test POST /telegram-webhook
    import bot
    bot._bot_ready_event.set()
    bot._global_app = MagicMock()
    bot._global_loop = MagicMock()
    bot._global_loop.is_closed.return_value = False
    bot._global_app.bot = MagicMock()

    handler.path = "/telegram-webhook"
    handler.send_response.reset_mock()
    handler.wfile = io.BytesIO()
    payload = b'{"update_id": 1}'
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)
    handler.do_POST()
    handler.send_response.assert_called_with(200)
    body_post = handler.wfile.getvalue()
    assert b"ok" in body_post


# ---------------------------------------------------------------------------
# 9. Bot Handler Registration Tests
# ---------------------------------------------------------------------------
def test_bot_handlers_registration():
    from bot import build_application
    app = build_application("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
    assert app is not None
    # Verify command handlers are registered
    registered_handlers = app.handlers.get(0, [])
    assert len(registered_handlers) >= 9


# ---------------------------------------------------------------------------
# 10. Webhook & Lifecycle Regression Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_start_command():
    """Verify /start handler resets session, sets state to AWAITING_JD, and sends welcome message."""
    import bot
    from unittest.mock import AsyncMock, MagicMock
    from telegram import Update, User, Chat, Message

    chat_id = 998877
    user = User(id=chat_id, first_name="Alice", is_bot=False)
    chat = Chat(id=chat_id, type="private")

    mock_msg = MagicMock(spec=Message)
    mock_msg.chat = chat
    mock_msg.from_user = user
    mock_msg.text = "/start"
    mock_msg.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.effective_chat = chat
    update.effective_message = mock_msg
    update.message = mock_msg

    context = MagicMock()
    context.bot.send_message = AsyncMock()

    await bot.start_command(update, context)

    # Verify session state
    case_id, case = bot.get_or_create_user_case(chat_id)
    assert case["state"] == "AWAITING_JD"

    # Verify reply message contents
    mock_msg.reply_text.assert_called_once()
    reply_text = mock_msg.reply_text.call_args[0][0]
    assert "Welcome to the ATS Resume Screening Bot." in reply_text
    assert "Please send the Job Description first." in reply_text


@pytest.mark.asyncio
async def test_start_resets_existing_session():
    """Verify /start resets old session even if previous JD and analyzed resumes exist."""
    import bot
    from unittest.mock import AsyncMock, MagicMock
    from telegram import Update, User, Chat, Message

    chat_id = 112233
    # Setup prior session with data
    old_case_id, _ = bot.get_or_create_user_case(chat_id)
    set_job_desc(old_case_id, "Old Senior Python Architect job description")
    add_resume(old_case_id, "Old_Candidate.pdf", "Senior Python Architect with 10 years experience")
    set_state(old_case_id, "ANALYZED")

    assert get_state(old_case_id) == "ANALYZED"
    assert len(get_resumes(old_case_id)) == 1

    # Now execute /start
    user = User(id=chat_id, first_name="Bob", is_bot=False)
    chat = Chat(id=chat_id, type="private")
    mock_msg = MagicMock(spec=Message)
    mock_msg.chat = chat
    mock_msg.reply_text = AsyncMock()

    update = MagicMock(spec=Update)
    update.effective_chat = chat
    update.effective_message = mock_msg
    update.message = mock_msg
    context = MagicMock()

    await bot.start_command(update, context)

    new_case_id, new_case = bot.get_or_create_user_case(chat_id)
    assert new_case_id != old_case_id
    assert new_case["state"] == "AWAITING_JD"
    assert get_resumes(new_case_id) == []
    assert get_job_desc(new_case_id) == ""


def test_webhook_receives_update():
    """Verify UnifiedHTTPHandler processes POST /telegram-webhook cleanly and safely."""
    import bot
    from bot import UnifiedHTTPHandler, _bot_ready_event
    from unittest.mock import MagicMock

    _bot_ready_event.set()
    bot._global_app = MagicMock()
    bot._global_loop = MagicMock()
    bot._global_loop.is_closed.return_value = False
    bot._global_app.bot = MagicMock()

    handler = UnifiedHTTPHandler.__new__(UnifiedHTTPHandler)
    handler.path = "/telegram-webhook"
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    payload = b'{"update_id": 10001, "message": {"message_id": 5, "chat": {"id": 12345}, "text": "/start"}}'
    handler.headers = {"Content-Length": str(len(payload))}
    handler.rfile = io.BytesIO(payload)

    handler.do_POST()
    handler.send_response.assert_called_with(200)
    body = handler.wfile.getvalue()
    assert b"ok" in body


def test_health_and_diagnostic_endpoints():
    """Verify GET /health and GET /diagnostic endpoints."""
    from bot import UnifiedHTTPHandler, _bot_ready_event, _webhook_diag_data
    from unittest.mock import MagicMock
    import json

    _bot_ready_event.set()
    _webhook_diag_data["url"] = "https://ats-bot-zqhn.onrender.com/telegram-webhook"
    _webhook_diag_data["status"] = "registered"
    _webhook_diag_data["pending_update_count"] = 0

    handler = UnifiedHTTPHandler.__new__(UnifiedHTTPHandler)
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    # GET /health -> {"status": "ok"}
    handler.path = "/health"
    handler.wfile = io.BytesIO()
    handler.do_GET()
    handler.send_response.assert_called_with(200)
    health_data = json.loads(handler.wfile.getvalue().decode())
    assert health_data["status"] == "ok"

    # GET /diagnostic -> reports safe webhook info without tokens
    handler.path = "/diagnostic"
    handler.wfile = io.BytesIO()
    handler.send_response.reset_mock()
    handler.do_GET()
    handler.send_response.assert_called_with(200)
    diag_data = json.loads(handler.wfile.getvalue().decode())
    assert diag_data["status"] == "ok"
    assert diag_data["bot_ready"] is True
    assert diag_data["webhook_url"] == "https://ats-bot-zqhn.onrender.com/telegram-webhook"
    assert "token" not in diag_data
    assert "TELEGRAM_BOT_TOKEN" not in diag_data


@pytest.mark.asyncio
async def test_webhook_processes_start():
    """Verify that an incoming webhook update triggers start_command and resets session."""
    import bot
    from unittest.mock import AsyncMock, patch
    from telegram import Update
    from telegram.ext import Application

    with patch("telegram.Bot._do_post", new=AsyncMock(return_value={"id": 123, "is_bot": True, "first_name": "ATS Bot", "username": "ats_bot"})):
        app = bot.build_application("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        await app.initialize()
        await app.start()

        chat_id = 554433
        update_data = {
            "update_id": 9001,
            "message": {
                "message_id": 1,
                "date": 1700000000,
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": chat_id, "first_name": "Tester", "is_bot": False},
                "text": "/start",
                "entities": [{"type": "bot_command", "offset": 0, "length": 6}],
            },
        }
        update = Update.de_json(update_data, app.bot)

        with patch("bot.safe_reply", new=AsyncMock()) as mock_safe_reply:
            await app.process_update(update)
            mock_safe_reply.assert_called_once()
            called_text = mock_safe_reply.call_args[0][1]
            assert "Welcome to the ATS Resume Screening Bot." in called_text
            assert "Please send the Job Description first." in called_text

        # Verify session state in CaseStore
        case_id, case = bot.get_or_create_user_case(chat_id)
        assert case["state"] == "AWAITING_JD"

        await app.stop()
        await app.shutdown()


@pytest.mark.asyncio
async def test_multiple_resume_workflow():
    """Verify complete multi-resume workflow: /start -> JD -> multi-resumes -> analysis."""
    import bot
    from services import case_store
    from unittest.mock import AsyncMock, MagicMock, patch

    chat_id = 778899

    # 1. /start
    case_id, _ = bot.reset_user_case(chat_id)
    case_store.set_state(case_id, "AWAITING_JD")
    assert case_store.get_state(case_id) == "AWAITING_JD"

    # 2. User submits JD
    target_jd = (
        "Senior Backend Engineer:\n"
        "- 5+ years experience with Python, FastAPI, and Docker\n"
        "- Experience with PostgreSQL and Redis\n"
        "- Experience with CI/CD and unit testing"
    )
    case_store.set_job_desc(case_id, target_jd)
    assert case_store.get_state(case_id) == "AWAITING_RESUMES"
    assert case_store.get_job_desc(case_id) == target_jd

    # 3. User uploads multiple resumes
    c1 = case_store.add_resume(
        case_id,
        "Alice_Python_Lead.pdf",
        "Senior Backend Engineer with 6 years Python, FastAPI, Docker, and PostgreSQL experience."
    )
    assert c1 == 1

    c2 = case_store.add_resume(
        case_id,
        "Bob_Frontend_Dev.pdf",
        "Frontend developer with React, TypeScript, HTML, CSS, and some Node.js experience."
    )
    assert c2 == 2

    resumes = case_store.get_resumes(case_id)
    assert len(resumes) == 2

    # 4. Run batch analysis with mock Gemini report
    mock_alice_report = {
        "verdict": "Excellent candidate for Senior Backend role.",
        "matched_skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
        "missing_skills": ["Redis"],
        "suggestions": ["Highlight Redis experience if available."],
        "calculated_score": 88,
        "role_alignment": "Strong Match",
    }
    mock_bob_report = {
        "verdict": "Frontend specialist with limited Python backend experience.",
        "matched_skills": ["Node.js"],
        "missing_skills": ["Python", "FastAPI", "Docker", "PostgreSQL", "Redis"],
        "suggestions": ["Transition into backend projects."],
        "calculated_score": 25,
        "role_alignment": "Moderate Match",
    }

    def mock_generate_report(resume_txt, jd_txt):
        if "Alice" in resume_txt or "Backend" in resume_txt:
            return mock_alice_report
        return mock_bob_report

    status_msg = MagicMock()
    status_msg.edit_text = AsyncMock()

    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message.reply_text = AsyncMock()
    context = MagicMock()

    with patch("bot.generate_report", side_effect=mock_generate_report):
        await bot.perform_batch_analysis(update, context, case_id, resumes, target_jd, status_msg)

    # Verify results in case store
    updated_case = case_store.get_case(case_id)
    analyzed_results = updated_case.get("analyzed_results", [])
    assert len(analyzed_results) == 2
    # Verify rankings (Alice rank 1, Bob rank 2)
    assert analyzed_results[0]["filename"] == "Alice_Python_Lead.pdf"
    assert analyzed_results[0]["ats_score"] > analyzed_results[1]["ats_score"]
    assert analyzed_results[1]["filename"] == "Bob_Frontend_Dev.pdf"

    # Pending resumes should be cleared after analysis, ready for next turn or questions
    assert len(case_store.get_resumes(case_id)) == 0
    case_store.delete_case(case_id)

