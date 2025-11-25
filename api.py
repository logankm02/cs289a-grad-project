from __future__ import annotations

"""Small FastAPI wrapper for the clustering functions.

Provides a lightweight JSON API for the React frontend:
- GET /api/health
- GET /api/clusters  -> run clustering synchronously and return results
- POST /api/clusters/run -> start clustering in background and return a task id
- GET /api/tasks/{task_id} -> check background task status/result

This keeps the existing CLI untouched and offers a minimal API layer to build a UI.
"""

import threading
import io
from contextlib import redirect_stdout, redirect_stderr
import uuid
from typing import Any, Dict, Optional, List

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cluster
import index as index_module


app = FastAPI(title="Email Clustering API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory task store (sufficient for local dev)
_TASKS: Dict[str, Dict[str, Any]] = {}
_LAST_CLUSTER_RESULT: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "email-cluster-api"}


@app.get("/api/clusters")
def get_clusters(user_email: Optional[str] = None, min_cluster_size: int = 5, max_threads: int = 1000):
    """Run clustering synchronously and return the results.

    For large datasets this will take time. Use `/api/clusters/run` to run async.
    """
    try:
        result = cluster.cluster_emails(
            user_email=user_email,
            collection_name=None,
            min_cluster_size=min_cluster_size,
            max_threads=max_threads,
            apply_labels=False,
            output_file=None,
        )
        global _LAST_CLUSTER_RESULT
        _LAST_CLUSTER_RESULT = result
    except Exception as exc:  # return a 400-like error to the frontend
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "ok", "result": result}


def _run_cluster_task(task_id: str, user_email: Optional[str], min_cluster_size: int, max_threads: int) -> None:
    try:
        _TASKS[task_id]["status"] = "running"
        res = cluster.cluster_emails(
            user_email=user_email,
            collection_name=None,
            min_cluster_size=min_cluster_size,
            max_threads=max_threads,
            apply_labels=False,
            output_file=None,
        )
        _TASKS[task_id]["status"] = "finished"
        _TASKS[task_id]["result"] = res
        global _LAST_CLUSTER_RESULT
        _LAST_CLUSTER_RESULT = res
    except Exception as exc:
        _TASKS[task_id]["status"] = "error"
        _TASKS[task_id]["error"] = str(exc)


@app.post("/api/clusters/run")
def run_clusters(background: BackgroundTasks, user_email: Optional[str] = None, min_cluster_size: int = 5, max_threads: int = 1000):
    """Start clustering in the background and return a task id.

    The task store is in-memory and only intended for local development.
    """
    task_id = uuid.uuid4().hex
    _TASKS[task_id] = {"status": "queued", "result": None}

    # Start a thread to run the clustering work so uvicorn isn't blocked.
    thread = threading.Thread(
        target=_run_cluster_task,
        args=(task_id, user_email, min_cluster_size, max_threads),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id, "status": "queued"}


def _run_index_task(
    task_id: str,
    user_email: str,
    max_threads: int,
    page_size: int,
    embed_batch: int,
    reset: bool,
    log_level: str,
) -> None:
    buf = io.StringIO()
    try:
        _TASKS[task_id]["status"] = "running"
        _TASKS[task_id]["result"] = None
        with redirect_stdout(buf), redirect_stderr(buf):
            res = index_module.index_gmail_threads(
                user_email=user_email,
                max_threads=max_threads,
                page_size=page_size,
                embed_batch=embed_batch,
                reset_first=reset,
            )
        _TASKS[task_id]["status"] = "finished"
        _TASKS[task_id]["result"] = {"summary": res, "log": buf.getvalue()}
    except Exception as exc:
        _TASKS[task_id]["status"] = "error"
        _TASKS[task_id]["error"] = str(exc)
        _TASKS[task_id]["log"] = buf.getvalue()


@app.post("/api/index/run")
def run_index(
    background: BackgroundTasks,
    user_email: str,
    max_threads: int = 100,
    page_size: int = 100,
    embed_batch: int = 64,
    reset: bool = False,
    log_level: str = "INFO",
):
    """Start indexing Gmail threads for a user in the background."""
    if not user_email:
        raise HTTPException(status_code=400, detail="user_email is required for indexing")
    task_id = uuid.uuid4().hex
    _TASKS[task_id] = {"status": "queued", "result": None, "kind": "index"}

    # ensure logger level aligns with request for this task
    cluster.configure_logging(log_level)

    thread = threading.Thread(
        target=_run_index_task,
        args=(task_id, user_email, max_threads, page_size, embed_batch, reset, log_level),
        daemon=True,
    )
    thread.start()
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = _TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/api/clusters/last")
def get_last_cluster():
    if _LAST_CLUSTER_RESULT is None:
        raise HTTPException(status_code=404, detail="no cluster result available")
    return {"status": "ok", "result": _LAST_CLUSTER_RESULT}


@app.get("/api/embeddings")
def get_embeddings(user_email: Optional[str] = None, max_threads: int = 1000) -> Dict[str, Any]:
    """Return 2D-projected coordinates (PCA) for thread embeddings and metadata.

    Requires `user_email` to identify the Chroma collection namespace.
    """
    if not user_email:
        raise HTTPException(status_code=400, detail="user_email query parameter is required")

    sanitized = cluster.hash_email_for_id(user_email)
    if not sanitized:
        raise HTTPException(status_code=400, detail="invalid user_email")

    try:
        collection = cluster._get_collection(sanitized)
        ordered_ids, vectors, metadatas = cluster._fetch_thread_vectors(collection, sanitized, max_threads)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to fetch embeddings: {exc}")

    if not ordered_ids:
        return {"status": "ok", "points": []}

    # build matrix in order
    matrix = np.vstack([vectors[tid] for tid in ordered_ids]).astype(np.float64)
    # center
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # PCA via SVD
    try:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2]
        coords = centered @ components.T
        xs = coords[:, 0].tolist()
        ys = coords[:, 1].tolist()
    except Exception:
        # fallback if SVD fails
        xs = [0.0] * matrix.shape[0]
        ys = [0.0] * matrix.shape[0]

    points: List[Dict[str, Any]] = []
    for i, tid in enumerate(ordered_ids):
        meta = metadatas.get(tid, {})
        points.append({
            "thread_id": tid,
            "x": xs[i],
            "y": ys[i],
            "metadata": meta,
        })

    return {"status": "ok", "points": points}
