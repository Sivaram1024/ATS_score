"""All Gemini interactions: the structured match report and the RAG-grounded chat ("AI Copilot")."""

import json
import logging
import re

import google.generativeai as genai

from config import Config

logger = logging.getLogger(__name__)

genai.configure(api_key=Config.GEMINI_API_KEY)

REPORT_PROMPT = """You are an exacting but fair ATS resume auditor.

Compare the RESUME against the JOB DESCRIPTION and respond with STRICT JSON ONLY —
no markdown fences, no commentary, no leading/trailing text. Follow this exact schema:

{{
  "verdict": "<one short sentence, max 15 words, plain assessment of fit>",
  "matched_skills": ["<short phrase, max 4 words>", ...],
  "missing_skills": ["<short phrase, max 4 words>", ...],
  "partial_matches": ["<short phrase, max 4 words>", ...],
  "suggestions": ["<short actionable phrase, max 8 words>", ...]
}}

Rules:
- 3 to 6 items per list where possible. Empty list if genuinely none apply.
- No duplicate items across lists.
- No sentences inside list items — short labels only.
- JSON must be valid and parseable.

RESUME:
{resume}

JOB DESCRIPTION:
{job_desc}
"""

CHAT_SYSTEM_INSTRUCTION = """You are the AI Career Copilot inside ResumeIQ, an ATS resume-analysis tool. \
A resume has already been scored against a job description, producing an ATS Score (BERT semantic \
similarity) and a structured match report.

You answer follow-up questions using a Retrieval-Augmented Generation (RAG) setup: for each \
question, the most relevant passages are retrieved from the resume and job description via \
embedding similarity and given to you below as RETRIEVED CONTEXT — you do not see the full \
documents, only the passages judged most relevant to this specific question, plus the report.

Answer using ONLY the retrieved context and the report. Be specific and reference actual content \
when relevant. Keep answers tight: 2-5 sentences unless the user explicitly asks for more detail \
or a list. Never invent experience that isn't in the retrieved context. If the retrieved context \
doesn't contain enough to answer confidently, say so plainly instead of guessing. If asked \
something outside the scope of this resume/job comparison, redirect politely back to the analysis.
"""


class GeminiServiceError(Exception):
    """Raised when Gemini can't produce a usable response."""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _empty_report(note: str) -> dict:
    return {
        "verdict": note,
        "matched_skills": [],
        "missing_skills": [],
        "partial_matches": [],
        "suggestions": [],
    }


def generate_report(resume: str, job_desc: str) -> dict:
    """Returns a structured dict; never raises to the caller (falls back to an empty-ish report)."""
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        prompt = REPORT_PROMPT.format(resume=resume[:12000], job_desc=job_desc[:6000])
        response = model.generate_content(prompt)
        cleaned = _strip_json_fences(response.text)
        data = json.loads(cleaned)

        return {
            "verdict": str(data.get("verdict", "")).strip(),
            "matched_skills": [str(x).strip() for x in data.get("matched_skills", [])][:8],
            "missing_skills": [str(x).strip() for x in data.get("missing_skills", [])][:8],
            "partial_matches": [str(x).strip() for x in data.get("partial_matches", [])][:8],
            "suggestions": [str(x).strip() for x in data.get("suggestions", [])][:8],
        }

    except json.JSONDecodeError:
        logger.exception("Gemini returned non-JSON report")
        return _empty_report("Report generation returned an unexpected format — try re-running the analysis.")
    except Exception:
        logger.exception("Gemini report generation failed")
        return _empty_report("AI feedback is temporarily unavailable — the similarity score above is still valid.")


def chat_reply(retrieved_context: str, report: dict, ats_score_pct: float, history: list, user_message: str) -> str:
    """
    retrieved_context: pre-formatted string of top-k RAG chunks for this specific question.
    ats_score_pct: the resume's ATS Score (0-100), so the Copilot can answer "why is my score X" directly.
    history: list of {"role": "user"|"model", "text": str}, oldest first.
    Returns the Copilot's reply text. Raises GeminiServiceError on failure.
    """
    try:
        model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
        )

        context_block = (
            f"ATS SCORE: {ats_score_pct}%\n\n"
            f"RETRIEVED CONTEXT (top-matching passages for this question):\n{retrieved_context}\n\n"
            f"MATCH REPORT:\n{json.dumps(report)}\n"
        )

        contents = [{"role": "user", "parts": [context_block]},
                    {"role": "model", "parts": ["Context loaded. Ask your question about the match."]}]

        for turn in history[-Config.MAX_CHAT_HISTORY:]:
            contents.append({"role": turn["role"], "parts": [turn["text"]]})

        contents.append({"role": "user", "parts": [user_message]})

        response = model.generate_content(contents)
        reply = (response.text or "").strip()
        if not reply:
            raise GeminiServiceError("Empty response from Gemini")
        return reply

    except Exception as exc:
        logger.exception("Gemini chat failed")
        raise GeminiServiceError("The AI Copilot is unavailable right now — please try again in a moment.") from exc
