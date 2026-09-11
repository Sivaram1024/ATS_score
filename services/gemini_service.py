"""All Gemini interactions: the structured match report and the RAG-grounded chat ("AI Copilot")."""

import json
import logging
import re

import google.generativeai as genai

from config import Config

logger = logging.getLogger(__name__)

genai.configure(api_key=Config.GEMINI_API_KEY)

REPORT_PROMPT = """You are an exacting, expert Applicant Tracking System (ATS) auditor and hiring manager.

Thoroughly analyze the candidate's RESUME against the target JOB DESCRIPTION.

Step 1: Check Domain & Role Match
- Does this resume target the same professional role and engineering/business field as the JD?
- If the resume is for an UNRELATED field (e.g. Sales or Graphic Designer applying for Data Science, or Nurse applying for Software Engineer), mark role_alignment as "Unrelated Domain" and keep calculated_score strictly between 0 and 20.

Step 2: Technical Skills & Key Competencies
- List verified skills explicitly present in the resume that match JD requirements as "matched_skills".
- List required tools/technologies in the JD that are MISSING or NOT demonstrated in the resume as "missing_skills".
- List adjacent or partially demonstrated skills as "partial_matches".

Step 3: Calculate the Honest ATS Match Score (0 to 100)
- 85 - 100: Strong Match — Candidate satisfies nearly all required core skills and experience.
- 65 - 84:  Good / Moderate Match — Core skills match, but candidate lacks a few tools or secondary requirements.
- 40 - 64:  Weak Match — Missing several essential technologies, partial career pivot.
- 0 - 39:   Poor Match / Unrelated — Does not satisfy fundamental prerequisites of the role.

Respond with STRICT JSON ONLY — no code blocks, no markdown fences, no extra commentary.
Follow this exact JSON schema:
{{
  "role_alignment": "<Strong Match | Moderate Match | Weak Match | Unrelated Domain>",
  "calculated_score": <integer 0 to 100>,
  "verdict": "<one concise sentence summarizing overall fit, max 20 words>",
  "matched_skills": ["<matched skill 1>", ...],
  "missing_skills": ["<missing skill 1>", ...],
  "partial_matches": ["<partial skill 1>", ...],
  "suggestions": [
    "<specific, actionable recommendation to improve this resume for this job>",
    ...
  ]
}}

RESUME:
{resume}

JOB DESCRIPTION:
{job_desc}
"""

CHAT_SYSTEM_INSTRUCTION = """You are the AI Career Copilot inside ResumeIQ, an ATS recruitment and resume-analysis assistant.
One or more candidate resumes have been evaluated against a job description, producing verified ATS scores and structured skill match reports.

You answer follow-up questions using Retrieval-Augmented Generation (RAG):
- If one resume was analyzed, provide specific feedback on how the candidate can improve, why certain skills are missing, or how to rephrase bullet points.
- If multiple resumes were analyzed, you can compare candidates side-by-side (e.g. who is stronger, why Rank 1 scored higher, what each candidate lacks).
- Answer using ONLY the retrieved context and the match reports. Be grounded and specific. Keep answers concise: 2-5 sentences unless asked for a detailed breakdown or list.
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
        "role_alignment": "Unknown",
        "calculated_score": 50,
        "verdict": note,
        "matched_skills": [],
        "missing_skills": [],
        "partial_matches": [],
        "suggestions": [],
    }


def compute_blended_ats_score(report: dict, bert_similarity: float) -> tuple[float, str]:
    """
    Computes an honest, defensible ATS score based on:
    1. Verified skill coverage (matched vs missing)
    2. Deep Gemini role & requirements audit
    3. Contextual semantic similarity (capped if skills don't match)
    Returns: (score_pct, rating_label)
    """
    matched = report.get("matched_skills", [])
    missing = report.get("missing_skills", [])
    partial = report.get("partial_matches", [])
    role_align = str(report.get("role_alignment", "")).strip().lower()
    gemini_score = report.get("calculated_score")

    n_matched = len(matched)
    n_missing = len(missing)
    n_partial = len(partial)
    total_skills = n_matched + n_missing + n_partial

    if total_skills > 0:
        skill_coverage_pct = ((n_matched * 1.0 + n_partial * 0.5) / total_skills) * 100.0
    else:
        skill_coverage_pct = 50.0

    bert_pct = max(0.0, min(100.0, bert_similarity * 100.0))

    # Check for completely unrelated domain or zero matched skills
    if "unrelated" in role_align or (n_matched == 0 and n_missing >= 2):
        score = min(20.0, float(gemini_score) if isinstance(gemini_score, (int, float)) else 12.0)
        return round(score, 1), "Unrelated Role"

    # Blended scoring: 60% Gemini expert rubric, 25% direct skill coverage, 15% BERT semantic
    if isinstance(gemini_score, (int, float)) and 0 <= gemini_score <= 100:
        score = (0.60 * float(gemini_score)) + (0.25 * skill_coverage_pct) + (0.15 * bert_pct)
    else:
        score = (0.75 * skill_coverage_pct) + (0.25 * bert_pct)

    # Sanity checks: penalize heavily if no core skills match
    if n_matched == 0 and score > 25.0:
        score = 25.0
    elif n_matched < 2 and n_missing >= 4 and score > 45.0:
        score = 45.0

    score = max(0.0, min(100.0, round(score, 1)))

    if score >= 85:
        rating = "Strong Match"
    elif score >= 70:
        rating = "Good Match"
    elif score >= 50:
        rating = "Moderate Match"
    elif score >= 35:
        rating = "Weak Match"
    else:
        rating = "Poor Match"

    return score, rating


def generate_report(resume: str, job_desc: str) -> dict:
    """Returns a structured dict with role alignment, score, and skill breakdown."""
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        prompt = REPORT_PROMPT.format(resume=resume[:14000], job_desc=job_desc[:7000])
        response = model.generate_content(prompt)
        cleaned = _strip_json_fences(response.text)
        data = json.loads(cleaned)

        raw_score = data.get("calculated_score")
        try:
            calculated_score = int(raw_score) if raw_score is not None else None
        except (ValueError, TypeError):
            calculated_score = None

        return {
            "role_alignment": str(data.get("role_alignment", "Moderate Match")).strip(),
            "calculated_score": calculated_score,
            "verdict": str(data.get("verdict", "")).strip(),
            "matched_skills": [str(x).strip() for x in data.get("matched_skills", [])][:10],
            "missing_skills": [str(x).strip() for x in data.get("missing_skills", [])][:10],
            "partial_matches": [str(x).strip() for x in data.get("partial_matches", [])][:10],
            "suggestions": [str(x).strip() for x in data.get("suggestions", [])][:8],
        }

    except json.JSONDecodeError:
        logger.exception("Gemini returned non-JSON report")
        return _empty_report("Report parsed with fallback structure — please re-evaluate.")
    except Exception:
        logger.exception("Gemini report generation failed")
        return _empty_report("AI evaluation feedback is temporarily unavailable.")


def chat_reply(retrieved_context: str, report: dict | list, ats_score_pct: float | None, history: list, user_message: str) -> str:
    try:
        model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
        )

        if isinstance(report, list):
            # Multi-candidate context
            summary_list = []
            for r in report:
                summary_list.append({
                    "candidate_file": r.get("filename"),
                    "ats_score": f"{r.get('ats_score')}%",
                    "rating": r.get("rating"),
                    "verdict": r.get("report", {}).get("verdict", ""),
                    "matched_skills": r.get("report", {}).get("matched_skills", []),
                    "missing_skills": r.get("report", {}).get("missing_skills", []),
                    "suggestions": r.get("report", {}).get("suggestions", []),
                })
            context_block = (
                f"CANDIDATES RANKING & EVALUATION SUMMARY:\n{json.dumps(summary_list, indent=2)}\n\n"
                f"RETRIEVED CONTEXT (relevant document excerpts):\n{retrieved_context}\n"
            )
        else:
            # Single candidate context
            context_block = (
                f"ATS MATCH SCORE: {ats_score_pct}%\n\n"
                f"ROLE ALIGNMENT: {report.get('role_alignment', 'N/A')}\n\n"
                f"RETRIEVED CONTEXT (top passages for this question):\n{retrieved_context}\n\n"
                f"MATCH REPORT (Skills & Suggestions):\n{json.dumps(report)}\n"
            )

        contents = [{"role": "user", "parts": [context_block]},
                    {"role": "model", "parts": ["Context loaded. Ask your question about the evaluation or comparison."]}]

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
