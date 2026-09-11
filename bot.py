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
from typing import Dict

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


async def safe_reply(message, text: str, **kwargs):
    """Reply to a message, safely falling back to plain text if formatting fails."""
    try:
        return await message.reply_text(text, **kwargs)
    except Exception as e:
        logger.warning(f"safe_reply failed with formatting: {e}. Falling back to clean plain text.")
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
            return await message.reply_text(plain)
        except Exception as ex2:
            logger.error(f"safe_reply fallback error: {ex2}")
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
    chat_id = update.effective_chat.id
    reset_user_case(chat_id)

    msg = (
        "🤖 *Welcome to ATS Score & AI Career Copilot!*\n\n"
        "I evaluate resumes against any Job Description using *verified skill matching*, "
        "*Sentence-BERT semantic alignment*, and *Google Gemini AI*.\n\n"
        "📋 *How to use me:*\n"
        "1️⃣ Send *1 or more Resumes as PDF files* 📎\n"
        "2️⃣ Send or paste the *Target Job Description* 📝\n"
        "3️⃣ I will calculate your honest ATS Score, matched competencies, and missing skills.\n"
        "4️⃣ If you upload multiple resumes, I will rank and compare them side-by-side!\n"
        "5️⃣ Ask follow-up questions to the AI Career Copilot anytime!\n\n"
        "📌 *Quick Commands:*\n"
        "• /sample - Instant test demo with benchmark profile\n"
        "• /report - Re-display your latest results\n"
        "• /export - Download analysis as a .txt file\n"
        "• /reset - Clear memory and start over\n"
        "• /help - Show full guide\n\n"
        "👉 *Send your Resume PDF to begin!*"
    )
    await safe_reply(update.message, msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed help message."""
    msg = (
        "💡 *ATS Score Bot Help & Workflow*\n\n"
        "• *Upload 1 or More Resumes:* Send `.pdf` files directly to the chat.\n"
        "• *Target Job Description:* Paste job requirements text.\n"
        "• *Reuse Previous JD:* Reply *\"use previous JD\"* to evaluate against your last job posting.\n"
        "• *Reuse Resumes:* Reply *\"use previous resume\"* to test with a new job description.\n"
        "• *Career Copilot:* After scoring, ask questions like:\n"
        "   _\"How can I improve my score?\"_\n"
        "   _\"Why is Docker missing?\"_\n"
        "   _\"Who is the best candidate and why?\"_\n\n"
        "• /reset - Clear memory & start fresh\n"
        "• /sample - Try an instant demo"
    )
    await safe_reply(update.message, msg)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the user's active session."""
    chat_id = update.effective_chat.id
    reset_user_case(chat_id)
    await safe_reply(
        update.message,
        "🔄 *Session reset!* Send your new resume PDF to start fresh.",
    )


async def sample_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Loads a benchmark candidate profile and target job description for quick demonstration."""
    chat_id = update.effective_chat.id
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

    case_store.add_resume(case_id, "Alex_Rivera_Senior_Dev.pdf", sample_resume)
    resumes = case_store.get_resumes(case_id)
    await perform_batch_analysis(update, context, case_id, resumes, sample_jd, status_msg)


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Re-display the latest analysis report."""
    chat_id = update.effective_chat.id
    case_id, case = get_or_create_user_case(chat_id)

    results = case.get("analyzed_results", [])
    if not results:
        await safe_reply(
            update.message,
            "⚠️ No analysis report found. Please upload a resume and job description first!",
        )
        return

    if len(results) == 1:
        top = results[0]
        report_msg = format_report_message(top["filename"], top["ats_score"], top["rating"], top["report"])
        await safe_reply(update.message, report_msg)
    else:
        ranking_text = format_ranking_message(results)
        await safe_reply(update.message, ranking_text)


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export the latest report as a downloadable .txt file."""
    chat_id = update.effective_chat.id
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


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle uploaded documents (PDFs). Supports 1 or more resumes."""
    document = update.message.document
    chat_id = update.effective_chat.id
    case_id, case = get_or_create_user_case(chat_id)

    filename = document.file_name or "Resume.pdf"
    if not filename.lower().endswith(".pdf"):
        await safe_reply(
            update.message,
            "⚠️ Please upload your resume in PDF format (.pdf).",
        )
        return

    status_msg = await safe_reply(
        update.message,
        f"⏳ Downloading and reading {filename}...",
    )

    try:
        tg_file = await context.bot.get_file(document.file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(out=buffer)
        buffer.seek(0)

        resume_text = extract_resume_text(buffer)
        count = case_store.add_resume(case_id, filename, resume_text)
        word_count = len(resume_text.split())

        case = case_store.get_case(case_id) or case
        prev_jd = case.get("previous_job_desc") or case.get("job_desc")

        tip_msg = ""
        if prev_jd:
            tip_msg = '\n\n💡 Tip: Reply "use previous JD" to evaluate against your previous job posting, or upload more resumes to compare them together!'
        else:
            tip_msg = "\n\n💡 Tip: You can upload additional resumes right now to compare multiple candidates side-by-side."

        if count == 1:
            msg = (
                f"✅ Resume Received: {filename} ({word_count} words extracted)\n\n"
                f"📝 Now please send or paste the Target Job Description to compare against!{tip_msg}"
            )
        else:
            msg = (
                f"✅ Added Resume {count}: {filename} ({word_count} words extracted)\n"
                f"📋 {count} resumes queued for evaluation!\n\n"
                f"📝 Send or paste the Target Job Description to analyze all {count} resumes together!{tip_msg}"
            )

        await safe_edit(status_msg, msg)

    except PDFExtractionError as exc:
        await safe_edit(status_msg, f"❌ PDF Extraction Failed: {str(exc)}")
    except Exception as exc:
        logger.exception("Error processing document")
        await safe_edit(status_msg, f"❌ Error processing PDF: {str(exc)}")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: JD input, previous JD/resume reuse commands, or Copilot chat."""
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    case_id, case = get_or_create_user_case(chat_id)

    # 1. Check for conversational shortcuts: "use previous JD" / "same JD"
    is_prev_jd_cmd = bool(re.search(r"^(use\s+)?(previous|same|last)\s+(jd|job\s*desc(ription)?)$", text, re.IGNORECASE))
    if is_prev_jd_cmd:
        prev_jd = case.get("previous_job_desc") or case.get("job_desc")
        if not prev_jd:
            await safe_reply(
                update.message,
                "⚠️ No previous Job Description found in memory. Please paste your Job Description text to begin.",
            )
            return

        resumes = case_store.get_resumes(case_id)
        if not resumes:
            await safe_reply(
                update.message,
                "⚠️ No resumes in queue. Please upload at least one Resume in PDF format first!",
            )
            return

        status_msg = await safe_reply(
            update.message,
            f"⏳ Evaluating {len(resumes)} resume(s) against your previous Job Description...",
        )
        await perform_batch_analysis(update, context, case_id, resumes, prev_jd, status_msg)
        return

    # 2. Check for conversational shortcut: "use previous resume" / "same resume"
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

    # 3. Check if text is a new Job Description or a follow-up Copilot question
    resumes = case_store.get_resumes(case_id)
    analyzed_results = case.get("analyzed_results", [])

    is_question = (
        "?" in text
        or bool(re.search(r"^(who|why|what|how|which|compare|can|is|tell|explain|suggest)", text, re.IGNORECASE))
        or len(text.split()) < 25
    )

    if analyzed_results and not resumes and is_question:
        status_msg = await safe_reply(update.message, "💬 Consulting AI Career Copilot...")
        await handle_chat_message(update, context, case_id, case, text, status_msg)
        return

    # Otherwise, treat as Job Description
    if resumes:
        if len(text) < 20:
            await safe_reply(
                update.message,
                "⚠️ That job description looks too short. Please paste the full job requirements to get an accurate score.",
            )
            return

        status_msg = await safe_reply(
            update.message,
            f"⏳ Job description received. Evaluating {len(resumes)} resume(s)...",
        )
        case_store.update_case(case_id, previous_job_desc=text)
        await perform_batch_analysis(update, context, case_id, resumes, text, status_msg)
        return

    # If no resumes yet, store JD if substantial
    if len(text.split()) >= 25:
        case_store.update_case(case_id, previous_job_desc=text)
        await safe_reply(
            update.message,
            f"📝 Job Description Stored ({len(text.split())} words).\n\n"
            "📄 Now please upload your Resume(s) as PDF file(s) to analyze against this JD!",
        )
    else:
        await safe_reply(
            update.message,
            "👋 Welcome! Please upload your Resume in PDF format to begin evaluation.",
        )


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


from http.server import HTTPServer, BaseHTTPRequestHandler


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK - ATS Score Telegram Bot is running 24/7.")

    def log_message(self, format, *args):
        pass


def run_health_server(port: int) -> None:
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        logger.info(f"Healthcheck HTTP server running on port {port}")
        server.serve_forever()
    except Exception as e:
        logger.warning(f"Failed to start healthcheck server on port {port}: {e}")


def main(in_thread: bool = False) -> None:
    token = Config.TELEGRAM_BOT_TOKEN
    if not token or token == "your-telegram-bot-token-here":
        print("[!] TELEGRAM_BOT_TOKEN is not configured.")
        return

    missing = Config.validate()
    if missing:
        print(f"[!] Warning: Missing environment variables: {', '.join(missing)}")

    print("[*] Starting ATS Score Telegram Bot...", flush=True)
    threading.Thread(target=get_model, daemon=True, name="bot-model-preloader").start()
    if not in_thread:
        port = int(os.getenv("PORT", "8080"))
        threading.Thread(target=run_health_server, args=(port,), daemon=True, name="bot-health-server").start()
        print(f"[*] Healthcheck HTTP server started on port {port}", flush=True)
    app = Application.builder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("sample", sample_command))
    app.add_handler(CommandHandler("report", report_command))
    app.add_handler(CommandHandler("export", export_command))

    # Document & Text handlers
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    print("[+] Telegram Bot is running! Press Ctrl+C to stop.", flush=True)
    if in_thread:
        app.run_polling(stop_signals=None)
    else:
        app.run_polling()


if __name__ == "__main__":
    main()
