"""
Lightweight server-side store for per-session case data (resume text, job
description, report, chat history) so we don't stuff large text into cookies.

This is an in-memory dict, which is fine for a single-process demo/portfolio
deployment. For production with multiple workers, swap this for Redis or a
database keyed the same way (by case_id) without changing the call sites.
"""

import threading
import uuid

_store: dict = {}
_lock = threading.Lock()


def create_case() -> str:
    case_id = uuid.uuid4().hex
    with _lock:
        _store[case_id] = {
            "resume_text": "",
            "job_desc": "",
            "resume_filename": "",
            "similarity": None,
            "report": None,
            "chat_history": [],
            "rag_chunks": [],
        }
    return case_id


def get_case(case_id: str) -> dict | None:
    with _lock:
        return _store.get(case_id)


def update_case(case_id: str, **fields) -> None:
    with _lock:
        if case_id in _store:
            _store[case_id].update(fields)


def append_chat(case_id: str, role: str, text: str) -> None:
    with _lock:
        if case_id in _store:
            _store[case_id]["chat_history"].append({"role": role, "text": text})


def delete_case(case_id: str) -> None:
    with _lock:
        _store.pop(case_id, None)
