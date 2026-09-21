"""
Thread-safe in-memory session registry supporting multi-resume queues,
persistent JD caching, comparative ranking history, and conversational memory.
"""

import threading
import uuid
from typing import Any, Dict, List, Optional

_registry: Dict[str, Dict[str, Any]] = {}
_store_lock = threading.Lock()


def create_case() -> str:
    """Creates a new evaluation session with unique session identifier."""
    session_id = uuid.uuid4().hex
    with _store_lock:
        _registry[session_id] = {
            "state": "AWAITING_JD",         # AWAITING_JD | AWAITING_RESUMES | ANALYZED
            "resumes": [],                  # List of {"filename": str, "text": str, "word_count": int}
            "job_desc": "",                 # Current active JD
            "previous_job_desc": "",        # Cached JD from previous analysis
            "active_job_desc": "",          # JD evaluated in current run
            "analyzed_results": [],         # List of analyzed candidate dicts
            "chat_history": [],             # Conversational Q&A turns
            "rag_chunks": [],               # Embedded vector document passages
            # Legacy compatibility pointers
            "resume_text": "",
            "resume_filename": "",
            "similarity": None,
            "report": None,
            "ats_score": None,
            "rating": None,
        }
    return session_id


def set_job_desc(case_id: str, jd_text: str) -> None:
    """Stores the active Job Description and transitions state to AWAITING_RESUMES."""
    with _store_lock:
        session = _registry.get(case_id)
        if session:
            session["job_desc"] = jd_text
            session["active_job_desc"] = jd_text
            session["previous_job_desc"] = jd_text
            session["state"] = "AWAITING_RESUMES"


def get_job_desc(case_id: str) -> str:
    """Retrieves the active or cached Job Description."""
    with _store_lock:
        session = _registry.get(case_id)
        if not session:
            return ""
        return session.get("job_desc") or session.get("active_job_desc") or session.get("previous_job_desc") or ""


def set_state(case_id: str, state: str) -> None:
    """Updates the workflow state of the session."""
    with _store_lock:
        session = _registry.get(case_id)
        if session:
            session["state"] = state


def get_state(case_id: str) -> str:
    """Retrieves the current workflow state of the session."""
    with _store_lock:
        session = _registry.get(case_id)
        if not session:
            return "AWAITING_JD"
        return session.get("state", "AWAITING_JD")


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves session record by ID."""
    with _store_lock:
        return _registry.get(case_id)


def update_case(case_id: str, **fields) -> None:
    """Updates specified fields in the active session."""
    with _store_lock:
        if case_id in _registry:
            _registry[case_id].update(fields)


def add_resume(case_id: str, filename: str, text: str) -> int:
    """
    Appends or updates a resume document in the queue.
    Returns total count of pending resumes in the session queue.
    """
    with _store_lock:
        session = _registry.get(case_id)
        if not session:
            return 0

        if "resumes" not in session:
            session["resumes"] = []

        words = len(text.split())
        updated = False
        for item in session["resumes"]:
            if item["filename"] == filename:
                item["text"] = text
                item["word_count"] = words
                updated = True
                break

        if not updated:
            session["resumes"].append({
                "filename": filename,
                "text": text,
                "word_count": words,
            })

        session["resume_text"] = text
        session["resume_filename"] = filename
        return len(session["resumes"])


def get_resumes(case_id: str) -> List[Dict[str, Any]]:
    """Retrieves all queued candidate resumes for the session."""
    with _store_lock:
        session = _registry.get(case_id)
        if not session:
            return []
        items = session.get("resumes", [])
        if not items and session.get("resume_text"):
            return [{
                "filename": session.get("resume_filename", "Candidate_Resume.pdf"),
                "text": session["resume_text"],
                "word_count": len(session["resume_text"].split()),
            }]
        return list(items)


def clear_pending_resumes(case_id: str) -> None:
    """Resets the pending resume queue while preserving cached JD and past analysis."""
    with _store_lock:
        if case_id in _registry:
            _registry[case_id]["resumes"] = []
            _registry[case_id]["resume_text"] = ""
            _registry[case_id]["resume_filename"] = ""


def append_chat(case_id: str, role: str, text: str) -> None:
    """Appends a dialogue turn to conversational history."""
    with _store_lock:
        if case_id in _registry:
            _registry[case_id]["chat_history"].append({"role": role, "text": text})


def delete_case(case_id: str) -> None:
    """Removes the session from the registry."""
    with _store_lock:
        _registry.pop(case_id, None)
