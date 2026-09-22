import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"
"""
ATS Score — Telegram Chatbot Interface
Combines Sentence-BERT semantic similarity on CPU, Google Gemini, and RAG Career Copilot.
Supports multi-resume upload, comparative ranking, and previous JD/resume reuse.
"""

import asyncio
import io
import json
import logging
import re
import sys
import threading
from typing import Any, Dict, Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from config import Config
from services import case_store, rag_service
from services.gemini_service import chat_reply, generate_report, compute_blended_ats_score, GeminiServiceError
from services.pdf_service import extract_resume_text, PDFExtractionError
from services.similarity_service import calculate_similarity, get_model

# Setup logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ats_telegram_bot")

def mask_id(val: Any) -> str:
    """Masks identifier (e.g. chat_id) for safe diagnostic logging."""
    s = str(val or "")
    if len(s) <= 4:
        return "****"
    return f"***{s[-4:]}"


# In-memory mapping from Telegram chat_id (int) -> case_id (str)
user_sessions: Dict[int, str] = {}
session_lock = threading.Lock()


def get_or_create_user_case(chat_id: int) -> tuple[str, dict]:
    with session_lock:
        case_id = user_sessions.get(chat_id)
        if not case_id or not case_store.get_case(case_id):
            case_id = case_store.create_case()
            user_sessions[chat_id] = case_id
        return case_id, case_store.get_case(case_id)


def reset_user_case(chat_id: int) -> tuple[str, dict]:
    with session_lock:
        old_id = user_sessions.pop(chat_id, None)
        if old_id:
            case_store.delete_case(old_id)
        case_id = case_store.create_case()
        user_sessions[chat_id] = case_id
        return case_id, case_store.get_case(case_id)


def escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 if needed."""
    if not text:
        return ""
    escape_chars = r"_*[]()~`>#+-=|{}.!\\"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", str(text))


async def safe_reply(message, text: str, context: Optional[Any] = None, chat_id: Optional[int] = None, **kwargs):
    """Reply to a message, safely falling back to plain text and send_message if formatting fails."""
    plain = (
        str(text)
        .replace(r"\\", "")
        .replace(r"\*", "")
        .replace(r"\_", "")
        .replace(r"\`", "")
        .replace(r"\~", "")
        .replace("*", "")
        .replace("_", "")
        .replace("`", "")
    )
    if message is not None:
        try:
            return await message.reply_text(text, **kwargs)
        except Exception as e:
            logger.warning(f"safe_reply failed with formatting: {e}. Falling back to clean plain text.")
            try:
                return await message.reply_text(plain)
            except Exception as ex2:
                logger.error(f"safe_reply message fallback failed: {ex2}")

    # Fallback to direct bot.send_message
    if context and hasattr(context, "bot") and chat_id is not None:
        try:
            return await context.bot.send_message(chat_id=chat_id, text=plain)
        except Exception as ex3:
            logger.error(f"safe_reply bot.send_message fallback failed: {ex3}")

    return None


async def safe_edit(msg, text: str, **kwargs):
    """Edit a message, safely falling back to plain text if formatting fails."""
    try:
        return await msg.edit_text(text, **kwargs)
    except Exception as e:
        logger.warning(f"safe_edit failed with formatting: {e}. Falling back to clean plain text.")
        try:
            plain = (
                str(text)
                .replace(r"\\", "")
                .replace(r"\*", "")
                .replace(r"\_", "")
                .replace(r"\`", "")
                .replace(r"\~", "")
                .replace("*", "")
                .replace("_", "")
                .replace("`", "")
            )
            return await msg.edit_text(plain)
        except Exception as ex2:
            logger.error(f"safe_edit fallback error: {ex2}")
            return None


try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
except ImportError:
    print("[!] python-telegram-bot is required. Run: pip install python-telegram-bot>=21.0")
    sys.exit(1)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message and instructions."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    masked_chat = mask_id(chat_id)
    logger.info("Executing /start command for chat %s", masked_chat)

    if chat_id is not None:
        reset_user_case(chat_id)
        case_id, _ = get_or_create_user_case(chat_id)
        case_store.set_state(case_id, "AWAITING_JD")
        logger.info("/start session reset completed for chat %s (state: AWAITING_JD)", masked_chat)

    msg = (
        "Welcome to the ATS Resume Screening Bot.\n\n"
        "Please send the Job Description first.\n\n"
        "📋 *Workflow:*\n"
        "1️⃣ Send or paste the target *Job Description* 📝\n"
        "2️⃣ Upload *1 or more Resumes as PDF files* 📎\n"
        "3️⃣ Send *Analyze* (or /analyze) to evaluate and rank all candidates 🏆\n"
        "4️⃣ Ask follow-up questions to the AI Career Copilot anytime! 💬\n\n"
        "📌 *Quick Commands:*\n"
        "• /sample - Instant demo with benchmark profile\n"
        "• /analyze - Run analysis on queued resumes\n"
        "• /report - Re-display latest results\n"
        "• /export - Download analysis as .txt\n"
        "• /reset - Clear memory and start over\n"
        "• /diagnostic - Check bot and webhook status\n"
        "• /help - Full guide and shortcuts\n\n"
        "👉 *Please paste your Job Description to begin!*"
    )
    msg_target = update.effective_message or update.message
    res = await safe_reply(msg_target, msg, context=context, chat_id=chat_id)
    if res:
        logger.info("Reply successfully sent for /start to chat %s", masked_chat)
    else:
        logger.warning("Failed to send /start reply to chat %s", masked_chat)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed help message."""
    chat_id = update.effective_chat.id
    logger.info("Received /help from chat %s", chat_id)
    msg = (
        "💡 *ATS Score Bot Guide & Workflow*\n\n"
        "1. *Send Job Description:* Paste job requirements text first.\n"
        "2. *Upload Resumes:* Send `.pdf` files directly to the chat (supports multiple candidates).\n"
        "3. *Analyze:* Send the message *\"Analyze\"* or use `/analyze` to score and rank candidates.\n"
        "4. *Conversational Shortcuts:*\n"
        "   • Reply *\"use previous JD\"* to evaluate new resumes against your last job posting.\n"
        "   • Reply *\"use previous resume\"* to test stored candidates against a new job description.\n"
        "5. *Career Copilot:* After scoring, ask questions like:\n"
        "   • _\"Why did Candidate A score higher?\"_\n"
        "   • _\"What is Candidate B missing?\"_\n"
        "   • _\"Compare Candidate A and Candidate B.\"_\n\n"
        "• /reset - Clear memory & start fresh\n"
        "• /sample - Try an instant benchmark demo\n"
        "• /export - Download full .txt audit report"
    )
    await safe_reply(update.message, msg)
    logger.info("Response sent to chat %s", chat_id)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the user's active session."""
    chat_id = update.effective_chat.id
    logger.info("Received /reset from chat %s", chat_id)
    reset_user_case(chat_id)
    case_id, _ = get_or_create_user_case(chat_id)
    case_store.set_state(case_id, "AWAITING_JD")
    await safe_reply(
        update.message,
        "🔄 *Session reset!* Send your Job Description first, then upload your resume PDF(s).",
    )
    logger.info("Response sent to chat %s", chat_id)


async def sample_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Loads a benchmark candidate profile and target job description for quick demonstration."""
    chat_id = update.effective_chat.id
    logger.info("Received /sample from chat %s", chat_id)
    case_id, case = reset_user_case(chat_id)

    status_msg = await safe_reply(update.message, "⏳ Loading benchmark Senior Python Engineer profile & running analysis...")

    sample_resume = (
        "Alex Rivera\n"
        "Senior Full-Stack Software Engineer | alex.rivera@example.com\n\n"
        "SUMMARY\n"
        "Results-driven software engineer with 6+ years of experience building scalable web applications. "
        "Proficient in Python, Flask, React, TypeScript, and Docker. Led development of high-throughput microservices handling 50k requests/sec.\n\n"
        "EXPERIENCE\n"
        "Lead Backend Engineer | TechCorp Inc. (2021 - Present)\n"
        "- Architected and maintained 15+ RESTful microservices using Python, Flask, and PostgreSQL.\n"
        "- Implemented asynchronous task pipelines using Celery and Redis, reducing background job latency by 45%.\n"
        "- Integrated Docker and CI/CD pipelines via GitHub Actions for automated testing and zero-downtime deployments.\n\n"
        "SKILLS\n"
        "Python, JavaScript, TypeScript, SQL, Flask, FastAPI, React, Node.js, Docker, Git, Redis, PostgreSQL, AWS"
    )

    sample_jd = (
        "Senior Python Engineer\n"
        "Requirements:\n"
        "- 5+ years of experience in Python and backend frameworks like Flask or FastAPI.\n"
        "- Strong knowledge of containerization with Docker and cloud deployments (AWS/GCP).\n"
        "- Experience with relational databases (PostgreSQL) and caching layers (Redis).\n"
        "- Familiarity with CI/CD workflows and automated testing.\n"
        "- Nice to have: Kubernetes experience."
    )

    case_store.set_job_desc(case_id, sample_jd)
    case_store.add_resume(case_id, "Alex_Rivera_Senior_Dev.pdf", sample_resume)
    resumes = case_store.get_resumes(case_id)
    logger.info("Analysis started for 1 sample resume in chat %s", chat_id)
    await perform_batch_analysis(update, context, case_id, resumes, sample_jd, status_msg)
    case_store.set_state(case_id, "ANALYZED")
    logger.info("Analysis completed for chat %s", chat_id)
    logger.info("Response sent to chat %s", chat_id)


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-display the latest analysis report."""
    chat_id = update.effective_chat.id
    logger.info("Received /report from chat %s", chat_id)
    case_id, case = get_or_create_user_case(chat_id)

    results = case.get("analyzed_results", [])
    if not results:
        await safe_reply(
            update.message,
            "⚠️ No analysis report found. Please provide a Job Description and upload at least one resume first!",
        )
        return

    if len(results) == 1:
        top = results[0]
        report_msg = format_report_message(top["filename"], top["ats_score"], top["rating"], top["report"])
        await safe_reply(update.message, report_msg)
    else:
        ranking_text = format_ranking_message(results)
        await safe_reply(update.message, ranking_text)
    logger.info("Response sent to chat %s", chat_id)


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export the latest report as a downloadable .txt file."""
    chat_id = update.effective_chat.id
    logger.info("Received /export from chat %s", chat_id)
    case_id, case = get_or_create_user_case(chat_id)

    results = case.get("analyzed_results", [])
    if not results:
        await safe_reply(update.message, "⚠️ No report to export. Run an evaluation first!")
        return

    content_lines = ["=" * 60, "ATS SCORE AUDIT REPORT", "=" * 60, ""]
    for r in results:
        content_lines.append(f"Candidate: {r.get('filename')}")
        content_lines.append(f"ATS Match Score: {r.get('ats_score')}% ({r.get('rating')})")
        rep = r.get("report", {})
        content_lines.append(f"Verdict: {rep.get('verdict')}")
        content_lines.append("Matched Skills: " + ", ".join(rep.get("matched_skills", [])))
        content_lines.append("Missing Skills: " + ", ".join(rep.get("missing_skills", [])))
        content_lines.append("Recommendations: " + "; ".join(rep.get("suggestions", [])))
        content_lines.append("-" * 60)

    export_bytes = "\n".join(content_lines).encode("utf-8")
    buffer = io.BytesIO(export_bytes)
    buffer.name = "ATS_Audit_Report.txt"
    await update.message.reply_document(document=buffer, filename="ATS_Audit_Report.txt")
    logger.info("Response sent to chat %s", chat_id)


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicit /analyze command handler."""
    chat_id = update.effective_chat.id
    logger.info("Received /analyze from chat %s", chat_id)
    case_id, case = get_or_create_user_case(chat_id)
    jd = case_store.get_job_desc(case_id)
    resumes = case_store.get_resumes(case_id)

    if not jd:
        await safe_reply(
            update.message,
            "⚠️ Please send your Target Job Description first before analyzing.",
        )
        return

    if not resumes:
        await safe_reply(
            update.message,
            "⚠️ No resumes in queue. Please upload at least one Resume in PDF format first!",
        )
        return

    logger.info("Analysis started for %d resumes in chat %s", len(resumes), chat_id)
    status_msg = await safe_reply(
        update.message,
        f"⏳ Evaluating {len(resumes)} resume(s) against the Job Description...",
    )
    await perform_batch_analysis(update, context, case_id, resumes, jd, status_msg)
    case_store.set_state(case_id, "ANALYZED")
    logger.info("Analysis completed for chat %s", chat_id)
    logger.info("Response sent to chat %s", chat_id)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle uploaded documents (PDFs). Supports 1 or more resumes."""
    document = update.message.document
    chat_id = update.effective_chat.id
    case_id, case = get_or_create_user_case(chat_id)

    filename = document.file_name or "Resume.pdf"
    logger.info("Received document: %s from chat %s", filename, chat_id)

    if not filename.lower().endswith(".pdf"):
        await safe_reply(
            update.message,
            "⚠️ Please upload your resume in PDF format (.pdf).",
        )
        return

    status_msg = await safe_reply(
        update.message,
        f"⏳ Reading and extracting {filename}...",
    )

    try:
        tg_file = await context.bot.get_file(document.file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(out=buffer)
        buffer.seek(0)

        resume_text = extract_resume_text(buffer)
        count = case_store.add_resume(case_id, filename, resume_text)
        word_count = len(resume_text.split())
        logger.info("Resume received: %s (%d words) for chat %s", filename, word_count, chat_id)

        jd = case_store.get_job_desc(case_id)

        if jd:
            if count == 1:
                msg = (
                    f"✅ *Resume 1 received:* `{filename}` ({word_count} words extracted).\n\n"
                    "📎 You can upload more resumes now, or send *Analyze* to begin candidate evaluation!"
                )
            else:
                msg = (
                    f"✅ *Resume {count} received:* `{filename}` ({word_count} words extracted).\n"
                    f"📋 *{count} resumes queued for evaluation.*\n\n"
                    "📎 Upload more resumes, or send *Analyze* to calculate rankings and ATS scores!"
                )
        else:
            if count == 1:
                msg = (
                    f"✅ *Resume 1 received:* `{filename}` ({word_count} words extracted).\n\n"
                    "📝 Now please send or paste the *Target Job Description* to evaluate against!"
                )
            else:
                msg = (
                    f"✅ *Resume {count} received:* `{filename}` ({word_count} words extracted).\n"
                    f"📋 *{count} resumes queued.*\n\n"
                    "📝 Please send or paste the *Target Job Description* to evaluate against!"
                )

        await safe_edit(status_msg, msg)
        logger.info("Response sent to chat %s", chat_id)

    except PDFExtractionError as exc:
        await safe_edit(status_msg, f"❌ PDF Extraction Failed: {str(exc)}")
    except Exception as exc:
        logger.exception("Error processing document")
        await safe_edit(status_msg, f"❌ Error processing PDF: {str(exc)}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: JD input, Analyze trigger, shortcuts, or Copilot chat."""
    text = (update.effective_message.text or "").strip() if update.effective_message else ""
    chat_id = update.effective_chat.id if update.effective_chat else None
    masked_chat = mask_id(chat_id)
    case_id, case = get_or_create_user_case(chat_id)

    # 0. Intercept "start" or "/start" command in plain text
    if text.lower() in ("/start", "start"):
        logger.info("Routing plain text '%s' to start_command for chat %s", text, masked_chat)
        return await start_command(update, context)

    # 1. Check for Analyze trigger ("Analyze", "analyze", "/analyze", "run analysis")
    is_analyze_cmd = bool(re.search(r"^(/?analyze|run\s*analysis|start\s*analysis|evaluate)$", text, re.IGNORECASE))
    if is_analyze_cmd:
        jd = case_store.get_job_desc(case_id)
        resumes = case_store.get_resumes(case_id)

        if not jd:
            await safe_reply(
                update.message,
                "⚠️ Please send the Target Job Description first before running analysis.",
            )
            return

        if not resumes:
            await safe_reply(
                update.message,
                "⚠️ No resumes in queue. Please upload at least one Resume in PDF format first!",
            )
            return

        logger.info("Analysis started for %d resumes in chat %s", len(resumes), chat_id)
        status_msg = await safe_reply(
            update.message,
            f"⏳ Evaluating {len(resumes)} resume(s) against the Job Description...",
        )
        await perform_batch_analysis(update, context, case_id, resumes, jd, status_msg)
        case_store.set_state(case_id, "ANALYZED")
        logger.info("Analysis completed for chat %s", chat_id)
        logger.info("Response sent to chat %s", chat_id)
        return

    # 2. Check for conversational shortcut: "use previous JD" / "same JD"
    is_prev_jd_cmd = bool(re.search(r"^(use\s+)?(previous|same|last)\s+(jd|job\s*desc(ription)?)$", text, re.IGNORECASE))
    if is_prev_jd_cmd:
        prev_jd = case.get("previous_job_desc") or case.get("job_desc")
        if not prev_jd:
            await safe_reply(
                update.message,
                "⚠️ No previous Job Description found in memory. Please paste your Job Description text to begin.",
            )
            return

        case_store.set_job_desc(case_id, prev_jd)
        resumes = case_store.get_resumes(case_id)
        if not resumes:
            await safe_reply(
                update.message,
                f"📝 Loaded previous Job Description ({len(prev_jd.split())} words).\n\n"
                "📄 Now upload your Resume PDF(s) and send *Analyze* to evaluate!",
            )
            return

        logger.info("Analysis started for %d resumes (previous JD) in chat %s", len(resumes), chat_id)
        status_msg = await safe_reply(
            update.message,
            f"⏳ Evaluating {len(resumes)} resume(s) against your previous Job Description...",
        )
        await perform_batch_analysis(update, context, case_id, resumes, prev_jd, status_msg)
        case_store.set_state(case_id, "ANALYZED")
        logger.info("Analysis completed for chat %s", chat_id)
        logger.info("Response sent to chat %s", chat_id)
        return

    # 3. Check for conversational shortcut: "use previous resume" / "same resume"
    is_prev_resume_cmd = bool(re.search(r"^(use\s+)?(previous|same|last)\s+resume(s)?$", text, re.IGNORECASE))
    if is_prev_resume_cmd:
        analyzed = case.get("analyzed_results", [])
        if analyzed:
            for r in analyzed:
                case_store.add_resume(case_id, r.get("filename", "Resume.pdf"), r.get("text", ""))
            await safe_reply(
                update.message,
                f"✅ Reloaded {len(analyzed)} previous resume(s). Send or paste the new Job Description to evaluate against!",
            )
        else:
            await safe_reply(
                update.message,
                "⚠️ No previous resumes found. Please upload your resume PDF to begin.",
            )
        return

    # 4. Check if text is a follow-up Copilot question
    resumes = case_store.get_resumes(case_id)
    analyzed_results = case.get("analyzed_results", [])

    is_question = (
        "?" in text
        or bool(re.search(r"^(who|why|what|how|which|compare|can|is|tell|explain|suggest)", text, re.IGNORECASE))
        or (len(analyzed_results) > 0 and len(text.split()) < 25)
    )

    if analyzed_results and not resumes and is_question:
        logger.info("Received Copilot question from chat %s", chat_id)
        status_msg = await safe_reply(update.message, "💬 Consulting AI Career Copilot...")
        await handle_chat_message(update, context, case_id, case, text, status_msg)
        logger.info("Response sent to chat %s", chat_id)
        return

    # 5. Otherwise, treat text as a Job Description
    if len(text.strip()) < 15:
        await safe_reply(
            update.message,
            "⚠️ That job description looks too short. Please provide a detailed Job Description (requirements, skills, responsibilities).",
        )
        return

    case_store.set_job_desc(case_id, text)
    words = len(text.split())
    logger.info("JD stored (%d words) for chat %s", words, chat_id)

    if resumes:
        await safe_reply(
            update.message,
            f"📝 *Job Description updated!* ({words} words)\n"
            f"📋 {len(resumes)} resume(s) in queue.\n\n"
            "👉 Send *Analyze* to evaluate all resumes, or upload more resumes.",
        )
    else:
        await safe_reply(
            update.message,
            f"✅ *Job Description received and stored!* ({words} words)\n\n"
            "📄 Now please upload your *Resume(s) as PDF files* one by one.\n"
            "Once uploaded, send *Analyze* to calculate rankings and ATS scores.",
        )
    logger.info("Response sent to chat %s", chat_id)


async def perform_batch_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, case_id: str, resumes: list[dict], job_desc: str, status_msg=None) -> None:
    """Evaluates 1 or more resumes against the given Job Description."""
    try:
        loop = asyncio.get_running_loop()
        num_resumes = len(resumes)

        if status_msg is None:
            status_msg = await safe_reply(update.message, f"⏳ Evaluating {num_resumes} resume(s)...")

        results = []
        for idx, r in enumerate(resumes, start=1):
            fn = r.get("filename", f"Resume_{idx}.pdf")
            r_text = r.get("text", "")

            # Step 1: Gemini deep audit
            report = await loop.run_in_executor(None, generate_report, r_text, job_desc)

            # Step 2: CPU BERT semantic similarity
            similarity = await loop.run_in_executor(None, calculate_similarity, r_text, job_desc)

            # Step 3: Compute honest, skill-grounded ATS score
            ats_score, rating = compute_blended_ats_score(report, similarity)

            results.append({
                "filename": fn,
                "text": r_text,
                "ats_score": ats_score,
                "rating": rating,
                "similarity": similarity,
                "report": report,
            })

        # Sort by ATS score descending (Rank 1 = best)
        results.sort(key=lambda x: x["ats_score"], reverse=True)

        # Step 4: Index multi-resume RAG chunks
        rag_chunks = await loop.run_in_executor(None, rag_service.build_multi_resume_chunks, resumes, job_desc)

        # Update case store
        case_store.update_case(
            case_id,
            active_job_desc=job_desc,
            previous_job_desc=job_desc,
            analyzed_results=results,
            rag_chunks=rag_chunks,
            ats_score=results[0]["ats_score"],
            rating=results[0]["rating"],
            report=results[0]["report"],
            resume_filename=results[0]["filename"],
            resume_text=results[0]["text"],
        )
        case_store.clear_pending_resumes(case_id)

        # Output Results
        if num_resumes == 1:
            top = results[0]
            formatted_report = format_report_message(top["filename"], top["ats_score"], top["rating"], top["report"])
            await safe_edit(status_msg, formatted_report)
        else:
            ranking_text = format_ranking_message(results)
            await safe_edit(status_msg, ranking_text)

            for idx, res in enumerate(results, start=1):
                report_msg = format_report_message(res["filename"], res["ats_score"], res["rating"], res["report"])
                await safe_reply(update.message, report_msg)

    except Exception as exc:
        logger.exception("Batch analysis failed")
        if status_msg:
            await safe_edit(status_msg, f"❌ Analysis failed: {str(exc)}")
        else:
            await safe_reply(update.message, f"❌ Analysis failed: {str(exc)}")


def format_ranking_message(results: list[dict]) -> str:
    """Formats comparative candidate rankings across multiple resumes."""
    lines = [
        "🏆 *ATS MULTI-RESUME CANDIDATE RANKINGS*",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📋 *Total Resumes Evaluated:* {len(results)}\n",
    ]
    medals = ["🥇", "🥈", "🥉"]
    for idx, r in enumerate(results, start=1):
        medal = medals[idx - 1] if idx <= 3 else f"#{idx}"
        fn = r.get("filename", f"Candidate {idx}")
        score = str(r.get("ats_score", 0.0))
        rating = r.get("rating", "Evaluated")
        rep = r.get("report", {})
        matched = rep.get("matched_skills", [])
        missing = rep.get("missing_skills", [])

        top_matched = ", ".join(matched[:3]) if matched else "None detected"
        top_missing = ", ".join(missing[:3]) if missing else "None detected"

        lines.append(f"{medal} *Rank {idx}:* `{fn}` — *{score}%* ({rating})")
        lines.append(f"   ✓ *Top Matches:* {top_matched}")
        lines.append(f"   ✗ *Key Gaps:* {top_missing}\n")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("👇 Detailed individual breakdown for each candidate is provided below!")
    return "\n".join(lines)


def format_report_message(filename: str, ats_score: float, rating: str, report: dict) -> str:
    verdict = report.get("verdict", "Analysis complete.")
    matched = report.get("matched_skills", [])
    missing = report.get("missing_skills", [])
    suggestions = report.get("suggestions", [])

    matched_list = "\n".join(f"  ✓ {s}" for s in matched) if matched else "  (none detected)"
    missing_list = "\n".join(f"  ✗ {s}" for s in missing) if missing else "  (none detected)"
    sugg_list = "\n".join(f"  • {s}" for s in suggestions) if suggestions else "  (none apply)"

    blocks = min(10, max(0, int(round(ats_score / 10))))
    gauge = "█" * blocks + "░" * (10 - blocks)
    example = missing[0] if missing else "Kubernetes"

    return (
        f"🎯 *ATS MATCH ANALYSIS*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📄 *File:* `{filename}`\n"
        f"📊 *ATS Match Score:* *{ats_score}%* — _{rating}_\n"
        f"`{gauge}`\n"
        f"_(Verified Skill & JD Alignment)_\n\n"
        f"📋 *Recruiter Verdict:*\n"
        f"{verdict}\n\n"
        f"✅ *Matched Skills ({len(matched)}):*\n"
        f"{matched_list}\n\n"
        f"❌ *Missing Skills / Keywords ({len(missing)}):*\n"
        f"{missing_list}\n\n"
        f"💡 *Actionable Recommendations:*\n"
        f"{sugg_list}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 *AI Career Copilot Active!*\n"
        f"Ask me any question grounded in your resume and target role, e.g.:\n"
        f"👉 _\"How can I improve my score?\"_\n"
        f"👉 _\"Why is {example} missing?\"_\n"
        f"👉 _\"What is the biggest change I should make?\"_"
    )


async def handle_chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE, case_id: str, case: dict, text: str, status_msg) -> None:
    """Answer follow-up questions using multi-candidate RAG and Gemini."""
    try:
        loop = asyncio.get_running_loop()
        chunks = case.get("rag_chunks", [])
        analyzed_results = case.get("analyzed_results", [])
        report = analyzed_results if analyzed_results else case.get("report", {})
        ats_score = case.get("ats_score", 0.0)

        top_chunks = await loop.run_in_executor(None, rag_service.retrieve, text, chunks)
        retrieved_context = rag_service.format_context(top_chunks)

        case_store.append_chat(case_id, "user", text)
        history = case.get("chat_history", [])

        reply = await loop.run_in_executor(
            None,
            chat_reply,
            retrieved_context,
            report,
            ats_score,
            history,
            text,
        )
        case_store.append_chat(case_id, "model", reply)

        formatted_reply = f"🤖 *Career Copilot:*\n\n" + reply
        await safe_edit(status_msg, formatted_reply)

    except GeminiServiceError as exc:
        await safe_edit(status_msg, f"⚠️ {str(exc)}")
    except Exception as exc:
        logger.exception("Chat reply failed")
        await safe_edit(status_msg, f"❌ Error: {str(exc)}")


async def diagnostic_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Safe diagnostic information command without exposing credentials."""
    chat_id = update.effective_chat.id if update.effective_chat else None
    masked_chat = mask_id(chat_id)
    logger.info("Executing /diagnostic command for chat %s", masked_chat)

    url = _webhook_diag_data.get("url") or "(none)"
    status = "Active" if _bot_ready_event.is_set() else "Initializing"
    pending = _webhook_diag_data.get("pending_update_count", 0)
    last_err_date = _webhook_diag_data.get("last_error_date") or "None"
    last_err_msg = _webhook_diag_data.get("last_error_message") or "None"

    diag_text = (
        "🛠️ *ATS Bot Diagnostics*\n\n"
        f"• *Status:* `{status}`\n"
        f"• *Webhook URL:* `{url}`\n"
        f"• *Pending Updates:* `{pending}`\n"
        f"• *Last Error Date:* `{last_err_date}`\n"
        f"• *Last Error Message:* `{last_err_msg}`"
    )
    msg_target = update.effective_message or update.message
    await safe_reply(msg_target, diag_text, context=context, chat_id=chat_id)


from http.server import HTTPServer, BaseHTTPRequestHandler

# Global references for async webhook dispatching and readiness tracking
_global_app: Optional[Application] = None
_global_loop: Optional[asyncio.AbstractEventLoop] = None
_bot_ready_event = threading.Event()
_webhook_diag_data: Dict[str, Any] = {
    "url": "",
    "status": "initializing",
    "pending_update_count": 0,
    "last_error_date": None,
    "last_error_message": None,
}


class UnifiedHTTPHandler(BaseHTTPRequestHandler):
    """
    Unified HTTP request handler for Render Cloud deployment:
    - GET /health, HEAD /health -> 200 OK (Cloud healthchecks)
    - GET /diagnostic, HEAD /diagnostic -> 200 OK (Safe webhook diagnostics)
    - POST /telegram-webhook, POST /webhook, POST / -> 200 OK (Telegram updates)
    """

    def do_GET(self):
        clean_path = self.path.split("?")[0].rstrip("/")
        if clean_path in ("/health", ""):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return

        if clean_path in ("/diagnostic", "/diagnostics", "/webhook-info"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            diag = {
                "status": "ok",
                "bot_ready": _bot_ready_event.is_set(),
                "webhook_url": _webhook_diag_data.get("url", ""),
                "webhook_status": _webhook_diag_data.get("status", "unknown"),
                "pending_update_count": _webhook_diag_data.get("pending_update_count", 0),
                "last_error_date": _webhook_diag_data.get("last_error_date"),
                "last_error_message": _webhook_diag_data.get("last_error_message"),
            }
            self.wfile.write(json.dumps(diag).encode("utf-8"))
            return

        # Default fallback for other GET requests
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def do_POST(self):
        """Processes incoming Telegram updates in Webhook mode."""
        clean_path = self.path.split("?")[0].rstrip("/")
        logger.info("Received HTTP POST on path: %s", clean_path or "/")

        content_len = int(self.headers.get("Content-Length", 0))
        if content_len <= 0:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            return

        # Read POST body
        try:
            post_data = self.rfile.read(content_len)
            data = json.loads(post_data.decode("utf-8"))
        except Exception as exc:
            logger.error("Failed to parse incoming webhook JSON: %s", exc)
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"invalid_json"}')
            return

        # If bot is still starting up (e.g. Render waking from sleep), wait up to 15 seconds
        if not _bot_ready_event.is_set():
            logger.info("Waiting up to 15s for Telegram bot application to become ready...")
            ready = _bot_ready_event.wait(timeout=15.0)
            if not ready:
                logger.warning("Bot application not ready after 15s timeout. Returning 503 so Telegram retries.")
                self.send_response(503)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(b'{"ok":false,"error":"bot_not_ready"}')
                return

        global _global_app, _global_loop
        if not _global_app or not _global_loop or _global_loop.is_closed():
            logger.error("Global application or event loop is not active. Returning 503.")
            self.send_response(503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"loop_unavailable"}')
            return

        try:
            update = Update.de_json(data, _global_app.bot)
            update_id = getattr(update, "update_id", "unknown")
            chat_id = None
            update_type = "unknown"
            if update.message:
                update_type = "message"
                chat_id = update.message.chat_id
            elif update.edited_message:
                update_type = "edited_message"
                chat_id = update.edited_message.chat_id
            elif update.callback_query:
                update_type = "callback_query"
                chat_id = update.callback_query.message.chat_id if update.callback_query.message else None

            logger.info(
                "Webhook update parsed: id=%s, type=%s, chat=%s. Dispatching to application...",
                update_id,
                update_type,
                mask_id(chat_id),
            )

            # Dispatch update directly through application handlers in the event loop
            future = asyncio.run_coroutine_threadsafe(
                _global_app.process_update(update),
                _global_loop,
            )

            def _log_future_result(fut):
                try:
                    fut.result()
                    logger.info("Successfully processed update id=%s", update_id)
                except Exception as ex:
                    logger.exception("Exception in update processing for id=%s: %s", update_id, ex)

            future.add_done_callback(_log_future_result)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        except Exception as exc:
            logger.exception("Error dispatching incoming webhook update: %s", exc)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):
        pass  # Silence routine ping logs


# Backward compatibility alias
HealthCheckHandler = UnifiedHTTPHandler


def run_health_server(port: int) -> None:
    try:
        server = HTTPServer(("0.0.0.0", port), UnifiedHTTPHandler)
        logger.info(f"Unified HTTP server running on port {port} (supporting GET/HEAD /health & POST /telegram-webhook)")
        server.serve_forever()
    except Exception as e:
        logger.warning(f"Failed to start HTTP server on port {port}: {e}")


async def post_init_callback(application: Application) -> None:
    """
    Hook called during app initialization: logs status and ensures clean state.
    """
    logger.info("Telegram bot initialized successfully")


async def check_webhook_status() -> None:
    """Inspects Telegram webhook status without exposing bot token."""
    token = Config.TELEGRAM_BOT_TOKEN
    if not token or token == "your-telegram-bot-token-here":
        print("[!] TELEGRAM_BOT_TOKEN is not configured in environment.")
        return

    from telegram import Bot

    try:
        bot_instance = Bot(token)
        info = await bot_instance.get_webhook_info()
        print("=" * 60)
        print("TELEGRAM UPDATE CONFIGURATION & WEBHOOK STATUS")
        print("=" * 60)
        print(f"Configured Webhook URL:  {info.url or '(none - bot uses polling)'}")
        print(f"Has Custom Certificate:  {info.has_custom_certificate}")
        print(f"Pending Update Count:    {info.pending_update_count}")
        print(f"Last Error Date:         {info.last_error_date or 'None'}")
        print(f"Last Error Message:      {info.last_error_message or 'None'}")
        print(f"Max Connections:         {info.max_connections or 'N/A'}")
        print(f"Active Mechanism:        {'WEBHOOK' if info.url else 'POLLING'}")
        print("=" * 60)
    except Exception as exc:
        print(f"[!] Error fetching webhook info: {exc}")


def build_application(token: str) -> Application:
    """Constructs Application and registers all command, text, and document handlers."""
    app = Application.builder().token(token).post_init(post_init_callback).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    logger.info("Telegram handler registered: /start")
    app.add_handler(CommandHandler("analyze", analyze_command))
    logger.info("Telegram handler registered: /analyze")
    app.add_handler(CommandHandler("help", help_command))
    logger.info("Telegram handler registered: /help")
    app.add_handler(CommandHandler("reset", reset_command))
    logger.info("Telegram handler registered: /reset")
    app.add_handler(CommandHandler("sample", sample_command))
    logger.info("Telegram handler registered: /sample")
    app.add_handler(CommandHandler("report", report_command))
    logger.info("Telegram handler registered: /report")
    app.add_handler(CommandHandler("export", export_command))
    logger.info("Telegram handler registered: /export")
    app.add_handler(CommandHandler("diagnostic", diagnostic_command))
    logger.info("Telegram handler registered: /diagnostic")
    app.add_handler(CommandHandler("status", diagnostic_command))
    logger.info("Telegram handler registered: /status")

    # Document & Text handlers
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    logger.info("Telegram handler registered: Document (PDF resumes)")
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    logger.info("Telegram handler registered: Text messages (JD, Analyze, Copilot)")

    return app


def main(in_thread: bool = False) -> None:
    if "--check-webhook" in sys.argv:
        asyncio.run(check_webhook_status())
        return

    # 1. Read environment variables
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "7860"))
    token = Config.TELEGRAM_BOT_TOKEN
    is_render = bool(os.getenv("RENDER") or os.getenv("RENDER_EXTERNAL_URL") or os.getenv("WEBHOOK_URL"))
    webhook_url = os.getenv("WEBHOOK_URL") or os.getenv("RENDER_EXTERNAL_URL") or Config.WEBHOOK_URL

    print(f"[*] Starting ATS Score Application on {host}:{port}...", flush=True)

    # 2. START HTTP HEALTH SERVER IMMEDIATELY ON 0.0.0.0:$PORT
    # Must start before any other initialization to satisfy Render port detection
    try:
        server = HTTPServer((host, port), UnifiedHTTPHandler)
        server_thread = threading.Thread(
            target=server.serve_forever, daemon=True, name="unified-http-server"
        )
        server_thread.start()
        print(f"[*] Unified HTTP server listening on {host}:{port} (Health: GET /health, Webhook: POST /telegram-webhook, Diagnostics: GET /diagnostic)", flush=True)
    except Exception as e:
        logger.warning(f"Could not bind HTTP server on {host}:{port}: {e}")

    if not token or token == "your-telegram-bot-token-here":
        print("[!] TELEGRAM_BOT_TOKEN is not configured in environment.")
        # Keep HTTP server alive so cloud healthchecks succeed even if token pending
        if not in_thread:
            try:
                import time
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                pass
        return

    missing = Config.validate()
    if missing:
        print(f"[!] Warning: Missing environment variables: {', '.join(missing)}")

    # 3. PRODUCTION WEBHOOK MODE (Render / Production Cloud)
    # Never run polling if in production webhook mode!
    if webhook_url or is_render or Config.ENABLE_TELEGRAM_BOT:
        clean_url = (webhook_url or Config.DEFAULT_PRODUCTION_URL).rstrip("/")
        if clean_url and not clean_url.startswith("http"):
            clean_url = f"https://{clean_url}"
        webhook_endpoint = f"{clean_url}/telegram-webhook"
        logger.info("Telegram webhook target configured: %s", clean_url)
        logger.info("Starting in PRODUCTION WEBHOOK mode on port %d", port)

        async def run_webhook_production():
            global _global_app, _global_loop, _webhook_diag_data
            _global_loop = asyncio.get_running_loop()
            app = build_application(token)
            _global_app = app

            # Initialize and start PTB Application
            await app.initialize()
            await app.start()
            logger.info("Telegram application initialized and started successfully")

            # Check and register webhook with Telegram API
            if webhook_endpoint:
                try:
                    current_info = await app.bot.get_webhook_info()
                    logger.info(
                        "Current Telegram webhook status: configured_url=%s, pending_count=%s",
                        current_info.url or "(none)",
                        current_info.pending_update_count,
                    )

                    # Only register if URL changed or missing; never drop pending updates!
                    if current_info.url != webhook_endpoint:
                        logger.info("Registering webhook URL: %s", webhook_endpoint)
                        await app.bot.set_webhook(
                            url=webhook_endpoint,
                            drop_pending_updates=False,
                            allowed_updates=Update.ALL_TYPES,
                        )
                        logger.info("Telegram webhook registration call completed")
                    else:
                        logger.info("Webhook already correctly registered as: %s", webhook_endpoint)

                    # Verify registration
                    verified_info = await app.bot.get_webhook_info()
                    _webhook_diag_data["url"] = verified_info.url or ""
                    _webhook_diag_data["status"] = "registered" if verified_info.url else "unregistered"
                    _webhook_diag_data["pending_update_count"] = verified_info.pending_update_count
                    _webhook_diag_data["last_error_date"] = str(verified_info.last_error_date) if verified_info.last_error_date else None
                    _webhook_diag_data["last_error_message"] = verified_info.last_error_message

                    logger.info(
                        "Verified webhook registration: url=%s, pending_count=%d",
                        verified_info.url,
                        verified_info.pending_update_count,
                    )
                    print(f"[+] Webhook registered and verified: {verified_info.url}", flush=True)

                except Exception as exc:
                    logger.error("Failed to register/verify webhook with Telegram API: %s", exc)
                    _webhook_diag_data["status"] = f"error: {str(exc)}"
            else:
                logger.warning("No public webhook endpoint available to register with Telegram.")
                _webhook_diag_data["status"] = "missing_endpoint"

            # Signal readiness so incoming webhook POST requests can be processed
            _bot_ready_event.set()
            print("[+] Telegram Bot is ACTIVE in WEBHOOK mode! Awaiting updates...", flush=True)

            # Keep process alive
            stop_event = asyncio.Event()
            await stop_event.wait()

        asyncio.run(run_webhook_production())

    # 4. LOCAL POLLING MODE (Only when NOT on Render and NO webhook URL is provided)
    else:
        logger.info("No WEBHOOK_URL / RENDER_EXTERNAL_URL found and not on Render: Falling back to POLLING mode on port %d", port)
        app = build_application(token)

        # Clear any stale webhook before polling to prevent 409 conflict
        async def clear_webhook_and_poll():
            try:
                await app.bot.delete_webhook(drop_pending_updates=True)
                logger.info("Cleared stale Telegram webhook for clean polling")
            except Exception as exc:
                logger.warning("Could not delete webhook: %s", exc)

        try:
            asyncio.run(clear_webhook_and_poll())
        except Exception:
            pass

        _bot_ready_event.set()
        logger.info("Telegram polling started")
        print("[+] Telegram Bot is running via POLLING! Press Ctrl+C to stop.", flush=True)
        if in_thread:
            app.run_polling(drop_pending_updates=True, stop_signals=None)
        else:
            app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

