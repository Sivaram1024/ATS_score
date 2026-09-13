"""
Google Gemini AI integration: Structured ATS match audit, domain alignment verification,
defensive blended scoring, and conversational Career Copilot Q&A.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import google.generativeai as genai

from config import Config

logger = logging.getLogger(__name__)

genai.configure(api_key=Config.GEMINI_API_KEY)

REPORT_PROMPT = """You are an exacting, expert Applicant Tracking System (ATS) auditor and technical recruiter.

Thoroughly analyze the candidate's RESUME against the target JOB DESCRIPTION.

Step 1: Role & Domain Alignment Check
- Evaluate whether the candidate's professional background aligns with the target role and domain.
- If the candidate is from an UNRELATED domain (e.g., Graphic Designer or Sales Rep applying for Machine Learning Engineer, or Chef applying for Backend Developer), set role_alignment to "Unrelated Domain" and calculated_score strictly between 0 and 20.

Step 2: Skill Coverage Analysis
- matched_skills: Verified competencies explicitly demonstrated in the resume matching the JD.
- missing_skills: Essential technologies, tools, or requirements in the JD absent from the resume.
- partial_matches: Adjacent or partially demonstrated competencies.

Step 3: Honest ATS Match Scoring (0 to 100)
- 85 - 100: Strong Match — Thoroughly satisfies core requirements, stack, and domain depth.
- 65 - 84:  Good / Moderate Match — Solid foundation with a few minor missing tools.
- 40 - 64:  Weak Match — Missing several essential technologies; substantial skill gaps.
- 0 - 39:   Poor Match / Unrelated — Does not satisfy baseline role prerequisites.

Respond with STRICT JSON ONLY. No markdown formatting, code fences, or extraneous text.
JSON Schema:
{{
  "role_alignment": "<Strong Match | Moderate Match | Weak Match | Unrelated Domain>",
  "calculated_score": <integer 0 to 100>,
  "verdict": "<one concise sentence summarizing overall fit, max 20 words>",
  "matched_skills": ["<skill 1>", ...],
  "missing_skills": ["<skill 1>", ...],
  "partial_matches": ["<skill 1>", ...],
  "suggestions": [
    "<actionable recommendation to improve fit for this role>",
    ...
  ]
}}

RESUME:
{resume}

JOB DESCRIPTION:
{job_desc}
"""

CHAT_SYSTEM_INSTRUCTION = """You are the AI Career Copilot, an expert ATS recruitment and resume-evaluation assistant.
You provide objective, grounded career feedback, resume audits, and comparative candidate analysis.

Guidelines:
- When evaluating a single candidate, clarify skill gaps, suggest specific resume enhancements, and justify the match score.
- When evaluating multiple candidates, compare strengths and weaknesses side-by-side (e.g. why Rank 1 outperformed others, what each candidate lacks).
- Anchor answers firmly in the verified document context and score report.
- Keep answers concise, direct, and professional (2-5 sentences unless detailed breakdown requested).
"""


class GeminiServiceError(Exception):
    """Raised when Gemini API request fails or returns unusable content."""
    pass


def _clean_json_output(payload: str) -> str:
    """Removes code fences or extraneous wrappers from JSON response."""
    text = payload.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _default_fallback_report(note: str) -> Dict[str, Any]:
    """Provides safe default report if AI parsing fails."""
    return {
        "role_alignment": "Moderate Match",
        "calculated_score": 50,
        "verdict": note,
        "matched_skills": [],
        "missing_skills": [],
        "partial_matches": [],
        "suggestions": [],
    }


def compute_blended_ats_score(report: Dict[str, Any], bert_similarity: float) -> Tuple[float, str]:
    """
    Computes a defensible ATS score out of 100 based on:
    1. Verified skill coverage (matched vs missing competencies)
    2. Deep Gemini role & requirements audit
    3. Sentence-BERT semantic similarity on CPU
    Penalizes unrelated domains heavily (<20%).
    Returns (score_pct, rating_label).
    """
    matched = report.get("matched_skills", [])
    missing = report.get("missing_skills", [])
    partial = report.get("partial_matches", [])
    role_alignment = str(report.get("role_alignment", "")).strip().lower()
    raw_gemini_score = report.get("calculated_score")

    n_matched = len(matched)
    n_missing = len(missing)
    n_partial = len(partial)
    total_skills = n_matched + n_missing + n_partial

    if total_skills > 0:
        skill_coverage_pct = ((n_matched * 1.0 + n_partial * 0.5) / total_skills) * 100.0
    else:
        skill_coverage_pct = 50.0

    bert_pct = max(0.0, min(100.0, bert_similarity * 100.0))

    # Unrelated role guardrail
    if "unrelated" in role_alignment or (n_matched == 0 and n_missing >= 2):
        score = min(20.0, float(raw_gemini_score) if isinstance(raw_gemini_score, (int, float)) else 12.0)
        return round(score, 1), "Unrelated Role"

    # Blended scoring: 60% LLM audit rubric, 25% skill coverage ratio, 15% BERT semantic
    if isinstance(raw_gemini_score, (int, float)) and 0 <= raw_gemini_score <= 100:
        score = (0.60 * float(raw_gemini_score)) + (0.25 * skill_coverage_pct) + (0.15 * bert_pct)
    else:
        score = (0.75 * skill_coverage_pct) + (0.25 * bert_pct)

    # Missing core skill penalties
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


def generate_report(resume: str, job_desc: str) -> Dict[str, Any]:
    """Generates structured candidate ATS evaluation against the target JD."""
    try:
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        prompt = REPORT_PROMPT.format(resume=resume[:14000], job_desc=job_desc[:7000])
        res = model.generate_content(prompt)
        cleaned = _clean_json_output(res.text)
        data = json.loads(cleaned)

        raw_score = data.get("calculated_score")
        try:
            score_val = int(raw_score) if raw_score is not None else None
        except (ValueError, TypeError):
            score_val = None

        return {
            "role_alignment": str(data.get("role_alignment", "Moderate Match")).strip(),
            "calculated_score": score_val,
            "verdict": str(data.get("verdict", "")).strip(),
            "matched_skills": [str(s).strip() for s in data.get("matched_skills", [])][:10],
            "missing_skills": [str(s).strip() for s in data.get("missing_skills", [])][:10],
            "partial_matches": [str(s).strip() for s in data.get("partial_matches", [])][:10],
            "suggestions": [str(s).strip() for s in data.get("suggestions", [])][:8],
        }

    except json.JSONDecodeError:
        logger.exception("Failed to parse Gemini response as JSON")
        return _default_fallback_report("Evaluation parsed with standard fallback format.")
    except Exception as exc:
        logger.exception("Gemini report generation failed: %s", exc)
        return _default_fallback_report("AI evaluation feedback is temporarily unavailable.")


def chat_reply(
    retrieved_context: str,
    report: Union[Dict[str, Any], List[Dict[str, Any]]],
    ats_score_pct: Optional[float],
    history: List[Dict[str, str]],
    user_message: str
) -> str:
    """Answers recruiter or candidate questions grounded in RAG document excerpts and score reports."""
    try:
        model = genai.GenerativeModel(
            Config.GEMINI_MODEL,
            system_instruction=CHAT_SYSTEM_INSTRUCTION,
        )

        if isinstance(report, list):
            summary = [
                {
                    "candidate_file": c.get("filename"),
                    "ats_score": f"{c.get('ats_score')}%",
                    "rating": c.get("rating"),
                    "verdict": c.get("report", {}).get("verdict", ""),
                    "matched_skills": c.get("report", {}).get("matched_skills", []),
                    "missing_skills": c.get("report", {}).get("missing_skills", []),
                    "suggestions": c.get("report", {}).get("suggestions", []),
                }
                for c in report
            ]
            context_block = (
                f"CANDIDATES RANKING & EVALUATION SUMMARY:\n{json.dumps(summary, indent=2)}\n\n"
                f"RETRIEVED CONTEXT (document passages):\n{retrieved_context}\n"
            )
        else:
            context_block = (
                f"ATS MATCH SCORE: {ats_score_pct}%\n\n"
                f"ROLE ALIGNMENT: {report.get('role_alignment', 'N/A')}\n\n"
                f"RETRIEVED CONTEXT (document passages):\n{retrieved_context}\n\n"
                f"MATCH REPORT:\n{json.dumps(report, indent=2)}\n"
            )

        messages = [
            {"role": "user", "parts": [context_block]},
            {"role": "model", "parts": ["Context successfully loaded. What question do you have about the candidate evaluations?"]},
        ]

        for turn in history[-Config.MAX_CHAT_HISTORY:]:
            messages.append({"role": turn["role"], "parts": [turn["text"]]})

        messages.append({"role": "user", "parts": [user_message]})

        resp = model.generate_content(messages)
        reply = (resp.text or "").strip()
        if not reply:
            raise GeminiServiceError("Empty response returned by Gemini")
        return reply

    except Exception as exc:
        logger.exception("Gemini career copilot interaction failed: %s", exc)
        raise GeminiServiceError("The AI Career Copilot is temporarily unavailable. Please retry in a moment.") from exc
