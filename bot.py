"""
ATS Score — Telegram Chatbot Interface
Combines Sentence-BERT semantic similarity, Google Gemini, and RAG Career Copilot.
"""

import asyncio
import io
import logging
import os
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
from services.gemini_service import chat_reply, generate_report, GeminiServiceError
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
    logger.error("python-telegram-bot is not installed. Run: pip install python-telegram-bot>=21.0")
    sys.exit(1)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message and instructions."""
    chat_id = update.effective_chat.id
    reset_user_case(chat_id)

    msg = (
        "👋 *Welcome to ATS Score AI Career Copilot\\!*\n\n"
        "I evaluate how well your resume matches any target job description using "
        "*Sentence\\-BERT semantic similarity*, *Google Gemini*, and *Retrieval\\-Augmented Generation \\(RAG\\)*\\.\n\n"
        "📌 *How to use me:*\n"
        "1️⃣ Send your *Resume as a PDF file* 📄 \\(or paste your resume text\\)\\.\n"
        "2️⃣ Send or paste the *Job Description* 📝\\.\n"
        "3️⃣ I will calculate your ATS Score, matched competencies, and missing skills\\.\n"
        "4️⃣ You can then ask me any follow\\-up questions right here in the chat\\!\n\n"
        "💡 *Commands:*\n"
        "• `/sample` \\- Try an instant demo with a benchmark profile\n"
        "• `/report` \\- View your latest analysis summary\n"
        "• `/export` \\- Download your report as a `.txt` file\n"
        "• `/reset` \\- Start a new evaluation\n"
        "• `/help` \\- Show this guide\n\n"
        "👉 *Send me your resume PDF or text to begin\\!*"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help guide."""
    msg = (
        "🤖 *ATS Score Bot Guide*\n\n"
        "• *Upload PDF:* Simply attach and send any `.pdf` resume\\.\n"
        "• *Paste Text:* Send resume or job description text directly\\.\n"
        "• *Chat Mode:* Once scored, send questions like:\n"
        "  _\"How can I improve my score?\"_\n"
        "  _\"Why is Kubernetes showing as missing?\"_\n"
        "  _\"What's the single biggest change I should make?\"_\n\n"
        "• `/sample` \\- Instant test analysis\n"
        "• `/report` \\- Show current match results\n"
        "• `/export` \\- Download `.txt` analysis report\n"
        "• `/reset` \\- Clear data and start over"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the user's active session."""
    chat_id = update.effective_chat.id
    reset_user_case(chat_id)
    await update.message.reply_text(
        "🔄 *Session reset\\!* Send your new resume PDF or text to start fresh\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def sample_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Loads a benchmark candidate profile and target job description for quick demonstration."""
    chat_id = update.effective_chat.id
    case_id, case = reset_user_case(chat_id)

    status_msg = await update.message.reply_text("⏳ Loading benchmark Senior Python Engineer profile & running analysis…")

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

    await perform_analysis(update, context, case_id, sample_resume, sample_jd, "Alex_Rivera_Senior_Python.pdf", status_msg)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle uploaded documents (specifically PDFs)."""
    document = update.message.document
    chat_id = update.effective_chat.id
    case_id, case = get_or_create_user_case(chat_id)

    filename = document.file_name or "Resume.pdf"
    if not filename.lower().endswith(".pdf"):
        await update.message.reply_text(
            "⚠️ Please upload your resume in *PDF format* \\(`.pdf`\\)\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    status_msg = await update.message.reply_text(f"⏳ Downloading and reading *{escape_md(filename)}*…", parse_mode=ParseMode.MARKDOWN_V2)

    try:
        tg_file = await context.bot.get_file(document.file_id)
        buffer = io.BytesIO()
        await tg_file.download_to_memory(out=buffer)
        buffer.seek(0)

        resume_text = extract_resume_text(buffer)
        case_store.update_case(case_id, resume_text=resume_text, resume_filename=filename)

        # If job description is already available, run analysis immediately
        if case.get("job_desc"):
            await status_msg.edit_text("⏳ Both resume and job description received. Running AI analysis…")
            await perform_analysis(update, context, case_id, resume_text, case["job_desc"], filename, status_msg)
        else:
            word_count = len(resume_text.split())
            await status_msg.edit_text(
                f"✅ *Resume Received:* `{escape_md(filename)}` \\({word_count} words extracted\\)\n\n"
                "👉 Now please send or paste the *Target Job Description* to compare against\\!",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
    except PDFExtractionError as exc:
        await status_msg.edit_text(f"❌ *PDF Extraction Failed:* {escape_md(str(exc))}", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as exc:
        logger.exception("Error processing document")
        await status_msg.edit_text(f"❌ Error processing PDF: {escape_md(str(exc))}", parse_mode=ParseMode.MARKDOWN_V2)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: either resume/JD ingestion or Copilot chat."""
    text = update.message.text.strip()
    chat_id = update.effective_chat.id
    case_id, case = get_or_create_user_case(chat_id)

    resume_text = case.get("resume_text")
    job_desc = case.get("job_desc")
    report = case.get("report")

    # State 1: No resume text yet
    if not resume_text:
        case_store.update_case(case_id, resume_text=text, resume_filename="Pasted_Resume.txt")
        word_count = len(text.split())
        await update.message.reply_text(
            f"✅ *Resume Text Stored* \\({word_count} words\\)\\.\n\n"
            "👉 Now send or paste the *Target Job Description* to compare against\\!",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    # State 2: Resume exists, but no Job Description yet
    if not job_desc or not report:
        if len(text) < 30:
            await update.message.reply_text(
                "⚠️ That job description looks too short\\. Please paste the full job posting to get an accurate score\\.",
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            return

        status_msg = await update.message.reply_text("⏳ Job description received\\. Computing ATS score & Gemini audit…", parse_mode=ParseMode.MARKDOWN_V2)
        await perform_analysis(update, context, case_id, resume_text, text, case.get("resume_filename", "Resume.pdf"), status_msg)
        return

    # State 3: Analysis complete -> User is asking a question to the RAG Career Copilot
    status_msg = await update.message.reply_text("💭 Consulting AI Career Copilot…")
    await handle_chat_message(update, context, case_id, case, text, status_msg)


async def perform_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, case_id: str, resume_text: str, job_desc: str, filename: str, status_msg) -> None:
    """Runs similarity, Gemini report, and RAG chunking."""
    try:
        loop = asyncio.get_running_loop()

        # Step 1: BERT similarity (run in executor to keep event loop unblocked)
        similarity = await loop.run_in_executor(None, calculate_similarity, resume_text, job_desc)
        ats_score = round(similarity * 100, 1)

        # Step 2: Gemini audit report
        report = await loop.run_in_executor(None, generate_report, resume_text, job_desc)

        # Step 3: RAG chunk indexing
        rag_chunks = await loop.run_in_executor(None, rag_service.build_and_embed_chunks, resume_text, job_desc)

        case_store.update_case(
            case_id,
            resume_text=resume_text,
            job_desc=job_desc,
            resume_filename=filename,
            similarity=similarity,
            report=report,
            rag_chunks=rag_chunks,
        )

        formatted_report = format_report_message(filename, ats_score, report)
        await status_msg.edit_text(formatted_report, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as exc:
        logger.exception("Analysis failed")
        await status_msg.edit_text(f"❌ Analysis failed: {escape_md(str(exc))}", parse_mode=ParseMode.MARKDOWN_V2)


def format_report_message(filename: str, ats_score: float, report: dict) -> str:
    verdict = report.get("verdict", "Analysis complete.")
    matched = report.get("matched_skills", [])
    missing = report.get("missing_skills", [])
    suggestions = report.get("suggestions", [])

    matched_list = "\n".join(f"  ✓ {escape_md(s)}" for s in matched) if matched else "  _(none detected)_"
    missing_list = "\n".join(f"  ✕ {escape_md(s)}" for s in missing) if missing else "  _(none detected)_"
    sugg_list = "\n".join(f"  • {escape_md(s)}" for s in suggestions) if suggestions else "  _(none apply)_"

    # Visual gauge meter
    blocks = int(ats_score // 10)
    gauge = "🟩" * blocks + "⬜" * (10 - blocks)

    return (
        f"🎯 *ATS MATCH ANALYSIS*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📄 *File:* `{escape_md(filename)}`\n"
        f"📊 *ATS Score:* *{ats_score}%*\n"
        f"{gauge}\n"
        f"_\\(BERT Semantic Similarity\\)_\n\n"
        f"🧠 *AI Verdict:*\n"
        f"_{escape_md(verdict)}_\n\n"
        f"✅ *Matched Skills \\({len(matched)}\\):*\n"
        f"{matched_list}\n\n"
        f"❌ *Missing Skills / Keywords \\({len(missing)}\\):*\n"
        f"{missing_list}\n\n"
        f"💡 *Actionable Recommendations:*\n"
        f"{sugg_list}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💬 *AI Career Copilot Active\\!*\n"
        f"Ask me any question grounded in your resume and target role, e\\.g\\.:\n"
        f"👉 _\"How can I improve my score?\"_\n"
        f"👉 _\"Why is {escape_md(missing[0] if missing else 'Kubernetes')} missing?\"_\n"
        f"👉 _\"What is the biggest change I should make?\"_"
    )


async def handle_chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE, case_id: str, case: dict, question: str, status_msg) -> None:
    """RAG-grounded chat response via Gemini."""
    try:
        loop = asyncio.get_running_loop()
        rag_chunks = case.get("rag_chunks", [])
        top_chunks = rag_service.retrieve(question, rag_chunks, top_k=5)
        retrieved_context = rag_service.format_context(top_chunks)

        ats_score_pct = round((case.get("similarity") or 0.0) * 100, 1)

        reply = await loop.run_in_executor(
            None,
            chat_reply,
            retrieved_context,
            case.get("report") or {},
            ats_score_pct,
            case.get("chat_history", []),
            question,
        )

        case_store.append_chat(case_id, "user", question)
        case_store.append_chat(case_id, "model", reply)

        formatted_reply = f"🤖 *Career Copilot:*\n\n{escape_md(reply)}"
        await status_msg.edit_text(formatted_reply, parse_mode=ParseMode.MARKDOWN_V2)

    except GeminiServiceError as exc:
        await status_msg.edit_text(f"⚠️ {escape_md(str(exc))}", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as exc:
        logger.exception("Chat reply failed")
        await status_msg.edit_text(f"❌ Copilot error: {escape_md(str(exc))}", parse_mode=ParseMode.MARKDOWN_V2)


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Resend the current match report."""
    chat_id = update.effective_chat.id
    _, case = get_or_create_user_case(chat_id)
    if not case.get("report"):
        await update.message.reply_text("ℹ️ No active analysis found. Send a resume PDF or `/sample` to begin!", parse_mode=ParseMode.MARKDOWN_V2)
        return

    ats_score = round((case.get("similarity") or 0.0) * 100, 1)
    msg = format_report_message(case.get("resume_filename", "Resume.pdf"), ats_score, case["report"])
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Export the report as a .txt file directly in Telegram."""
    chat_id = update.effective_chat.id
    _, case = get_or_create_user_case(chat_id)
    report = case.get("report")

    if not report:
        await update.message.reply_text("ℹ️ No active analysis found. Run an evaluation first!", parse_mode=ParseMode.MARKDOWN_V2)
        return

    def fmt(items):
        return "\n".join(f"- {item}" for item in items) if items else "- (none)"

    ats_score = round((case.get("similarity") or 0.0) * 100, 1)
    filename = case.get("resume_filename", "Resume.pdf")

    body = f"""RESUMEIQ / ATS SCORE — AI RESUME ANALYSIS REPORT
Resume File: {filename}
ATS Score (BERT Semantic Similarity): {ats_score}%

AI VERDICT:
{report.get('verdict', '')}

MATCHED SKILLS:
{fmt(report.get('matched_skills', []))}

MISSING SKILLS / KEYWORDS:
{fmt(report.get('missing_skills', []))}

AI RECOMMENDATIONS:
{fmt(report.get('suggestions', []))}
"""
    bio = io.BytesIO(body.encode("utf-8"))
    bio.name = f"ATS_Score_Report_{chat_id}.txt"
    bio.seek(0)

    await update.message.reply_document(
        document=bio,
        filename=f"ATS_Score_Report.txt",
        caption="📄 Here is your full downloadable ATS Analysis Report!",
    )


def escape_md(text: str) -> str:
    """Escape Telegram MarkdownV2 reserved characters."""
    special = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special:
        text = text.replace(char, f"\\{char}")
    return text


def main() -> None:
    token = Config.TELEGRAM_BOT_TOKEN
    if not token or token == "your-telegram-bot-token-here":
        print("\n" + "=" * 65)
        print("[!] TELEGRAM_BOT_TOKEN IS NOT CONFIGURED!")
        print("=" * 65)
        print("To connect your Telegram bot:")
        print("1. Open Telegram and search for @BotFather (https://t.me/BotFather)")
        print("2. Send /newbot and follow instructions to get your Bot Token.")
        print("3. Add the token to your .env file:")
        print("   TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRstuVWXyz")
        print("4. Restart python bot.py")
        print("=" * 65 + "\n")
        sys.exit(1)

    missing = Config.validate()
    if missing:
        print(f"[!] Warning: Missing environment variables: {', '.join(missing)}")

    print("[*] Starting ATS Score Telegram Bot...", flush=True)
    threading.Thread(target=get_model, daemon=True, name="bot-model-preloader").start()
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
    app.run_polling()


if __name__ == "__main__":
    main()
