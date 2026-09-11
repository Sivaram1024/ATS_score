"""
In-memory session store supporting multi-resume uploads, JD caching, and ranking history.
"""

import threading
import uuid

_store: dict = {}
_lock = threading.Lock()


def create_case() -> str:
    case_id = uuid.uuid4().hex
    with _lock:
        _store[case_id] = {
            "resumes": [],                  # List of {"filename": str, "text": str, "word_count": int}
            "previous_job_desc": "",        # Persistent JD from last analysis
            "active_job_desc": "",          # JD used in the current run
            "analyzed_results": [],         # List of analyzed candidate dicts
            "chat_history": [],
            "rag_chunks": [],
            # Legacy compatibility fields:
            "resume_text": "",
            "resume_filename": "",
            "similarity": None,
            "report": None,
            "ats_score": None,
            "rating": None,
        }
    return case_id


def get_case(case_id: str) -> dict | None:
    with _lock:
        return _store.get(case_id)


def update_case(case_id: str, **fields) -> None:
    with _lock:
        if case_id in _store:
            _store[case_id].update(fields)


def add_resume(case_id: str, filename: str, text: str) -> int:
    """Adds or updates a resume in the current session's queue. Returns total queue count."""
    with _lock:
        case = _store.get(case_id)
        if not case:
            return 0
        if "resumes" not in case:
            case["resumes"] = []

        word_count = len(text.split())
        # Replace if same filename exists, else append
        found = False
        for r in case["resumes"]:
            if r["filename"] == filename:
                r["text"] = text
                r["word_count"] = word_count
                found = True
                break
        if not found:
            case["resumes"].append({
                "filename": filename,
                "text": text,
                "word_count": word_count,
            })

        # Update legacy single-resume pointers to latest
        case["resume_text"] = text
        case["resume_filename"] = filename
        return len(case["resumes"])


def get_resumes(case_id: str) -> list[dict]:
    """Returns the list of pending resumes in the queue."""
    with _lock:
        case = _store.get(case_id)
        if not case:
            return []
        res = case.get("resumes", [])
        if not res and case.get("resume_text"):
            return [{"filename": case.get("resume_filename", "Resume.pdf"), "text": case["resume_text"], "word_count": len(case["resume_text"].split())}]
        return list(res)


def clear_pending_resumes(case_id: str) -> None:
    """Clears pending resumes after analysis while retaining previous_job_desc and results."""
    with _lock:
        if case_id in _store:
            _store[case_id]["resumes"] = []


def append_chat(case_id: str, role: str, text: str) -> None:
    with _lock:
        if case_id in _store:
            _store[case_id]["chat_history"].append({"role": role, "text": text})


def delete_case(case_id: str) -> None:
    with _lock:
        _store.pop(case_id, None)
