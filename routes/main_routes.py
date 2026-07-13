import logging

from flask import (
    Blueprint, render_template, request, redirect, url_for,
    session, flash, jsonify, Response
)

from config import Config
from services.pdf_service import extract_resume_text, allowed_file, PDFExtractionError
from services.similarity_service import calculate_similarity
from services.gemini_service import generate_report, chat_reply, GeminiServiceError
from services import case_store, rag_service

logger = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__)


def _current_case():
    case_id = session.get("case_id")
    if not case_id:
        return None, None
    case = case_store.get_case(case_id)
    if not case:
        return None, None
    return case_id, case


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    job_desc = (request.form.get("job_desc") or "").strip()

    if not resume_file or resume_file.filename == "":
        flash("Please attach a resume PDF.", "error")
        return redirect(url_for("main.index"))

    if not allowed_file(resume_file.filename, Config.ALLOWED_EXTENSIONS):
        flash("Only PDF resumes are accepted right now.", "error")
        return redirect(url_for("main.index"))

    if len(job_desc) < 30:
        flash("Paste the full job description — that was too short to compare against.", "error")
        return redirect(url_for("main.index"))

    try:
        resume_text = extract_resume_text(resume_file)
    except PDFExtractionError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.index"))

    try:
        similarity = calculate_similarity(resume_text, job_desc)
    except Exception:
        logger.exception("Similarity calculation failed")
        flash("Something went wrong scoring the resume. Please try again.", "error")
        return redirect(url_for("main.index"))

    report = generate_report(resume_text, job_desc)

    try:
        rag_chunks = rag_service.build_and_embed_chunks(resume_text, job_desc)
    except Exception:
        logger.exception("RAG chunk embedding failed")
        rag_chunks = []

    case_id = case_store.create_case()
    case_store.update_case(
        case_id,
        resume_text=resume_text,
        job_desc=job_desc,
        resume_filename=resume_file.filename,
        similarity=similarity,
        report=report,
        rag_chunks=rag_chunks,
    )
    session["case_id"] = case_id
    return redirect(url_for("main.results"))


@main_bp.route("/results")
def results():
    case_id, case = _current_case()
    if not case:
        flash("Start a new analysis to see results.", "error")
        return redirect(url_for("main.index"))

    report = case["report"] or {}
    total_flagged = (
        len(report.get("matched_skills", []))
        + len(report.get("missing_skills", []))
        + len(report.get("partial_matches", []))
    )
    coverage = (
        round(len(report.get("matched_skills", [])) / total_flagged * 100)
        if total_flagged else None
    )

    suggested_questions = _build_suggested_questions(report)

    return render_template(
        "results.html",
        resume_filename=case["resume_filename"],
        similarity_pct=round(case["similarity"] * 100, 1),
        coverage_pct=coverage,
        report=report,
        chat_history=case["chat_history"],
        suggested_questions=suggested_questions,
    )


def _build_suggested_questions(report: dict) -> list:
    """Generates a few starter questions grounded in this specific report,
    so the RAG chat isn't a blank box for users who don't know what to ask."""
    questions = ["How can I improve my ATS score?"]

    missing = report.get("missing_skills") or []
    if missing:
        questions.append(f"Why is \"{missing[0]}\" showing up as missing?")

    partial = report.get("partial_matches") or []
    if partial:
        questions.append(f"Why is \"{partial[0]}\" only a partial match?")

    matched = report.get("matched_skills") or []
    if matched:
        questions.append(f"Which part of my resume matched \"{matched[0]}\"?")

    questions.append("What's the single biggest change I should make?")
    return questions[:4]


@main_bp.route("/chat", methods=["POST"])
def chat():
    case_id, case = _current_case()
    if not case:
        return jsonify({"error": "No active analysis. Run a scan first."}), 400

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Type a question first."}), 400
    if len(message) > 800:
        return jsonify({"error": "Keep questions under 800 characters."}), 400

    top_chunks = rag_service.retrieve(message, case.get("rag_chunks", []), top_k=5)
    retrieved_context = rag_service.format_context(top_chunks)

    try:
        reply = chat_reply(
            retrieved_context=retrieved_context,
            report=case["report"] or {},
            ats_score_pct=round(case["similarity"] * 100, 1),
            history=case["chat_history"],
            user_message=message,
        )
    except GeminiServiceError as exc:
        return jsonify({"error": str(exc)}), 502

    case_store.append_chat(case_id, "user", message)
    case_store.append_chat(case_id, "model", reply)
    return jsonify({"reply": reply})


@main_bp.route("/export")
def export():
    case_id, case = _current_case()
    if not case:
        flash("Start a new analysis first.", "error")
        return redirect(url_for("main.index"))

    report = case["report"] or {}

    def fmt(items):
        return "\n".join(f"- {item}" for item in items) if items else "- (none)"

    body = f"""RESUMEIQ — AI RESUME ANALYSIS REPORT
Resume file: {case['resume_filename']}
ATS Score (semantic similarity, BERT): {round(case['similarity'] * 100, 1)}%

AI VERDICT
{report.get('verdict', '')}

MATCHED SKILLS
{fmt(report.get('matched_skills', []))}

MISSING SKILLS / KEYWORDS
{fmt(report.get('missing_skills', []))}

PARTIAL MATCHES
{fmt(report.get('partial_matches', []))}

AI RECOMMENDATIONS
{fmt(report.get('suggestions', []))}
"""
    return Response(
        body,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=resumeiq_ats_report.txt"},
    )


@main_bp.route("/reset")
def reset():
    case_id = session.pop("case_id", None)
    if case_id:
        case_store.delete_case(case_id)
    return redirect(url_for("main.index"))
