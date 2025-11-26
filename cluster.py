# -*- coding: utf-8 -*-
"""Cluster Gmail threads stored in a local ChromaDB collection."""

import argparse
import base64
import hashlib
import logging
import os
import re
import sys
import time
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.version_info < (3, 10):
    raise RuntimeError(
        "Python 3.10 or higher is required to run these Gmail tools. "
        f"Current interpreter: {sys.version.split()[0]}"
    )

import hdbscan
import numpy as np
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

try:  # Optional Gemini support
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore

try:  # Optional OpenAI embeddings
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:  # Optional local embeddings
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

try:
    import chromadb
except ImportError as import_error:  # pragma: no cover - runtime guard
    chromadb = None  # type: ignore
    CHROMA_IMPORT_ERROR = import_error
else:
    CHROMA_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Logging & configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("gmail_tools")


def configure_logging(level: str = "INFO") -> None:
    """Configure the shared logger with the requested level."""
    numeric_level = getattr(logging, level.upper(), None)
    warn_message = None
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        warn_message = f"Unrecognized log level '{level}'; defaulting to INFO"
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(numeric_level)
    logger.propagate = False
    if warn_message:
        logger.warning(warn_message)


configure_logging()

if load_dotenv:
    dotenv_path = os.getenv("CS289A_ENV_FILE", ".env")
    load_dotenv(dotenv_path=dotenv_path, override=False)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]

GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", "token.json")


def _resolve_client_secret_path() -> str:
    """Prefer desktop credentials unless explicitly overridden."""
    env_override = os.getenv("GMAIL_CLIENT_SECRET")
    if env_override:
        return env_override

    for candidate in (
        "desktop_credentials.json",
        "credentials.json",
        "web_credentials.json",
    ):
        if os.path.exists(candidate):
            return candidate

    return "credentials.json"


GMAIL_CLIENT_SECRET = _resolve_client_secret_path()


def _write_cluster_summary(
    output_path: str,
    clusters: Dict[str, Dict[str, Any]],
    noise_threads: List[Dict[str, Any]],
) -> None:
    """Persist a readable cluster summary with subjects grouped by label."""
    logger.info(
        "Writing cluster summary to %s (%d labeled clusters, %d noise threads)",
        output_path,
        len(clusters),
        len(noise_threads),
    )
    try:
        lines: List[str] = []
        timestamp = datetime.now().isoformat()
        lines.append(f"Email Cluster Summary ({timestamp})")
        lines.append("=" * 80)

        if not clusters:
            lines.append("No clusters discovered.")
        else:
            sorted_clusters = sorted(
                clusters.items(), key=lambda item: item[1].get("cluster_id", 0)
            )
            for cluster_label, info in sorted_clusters:
                lines.append("")
                lines.append(f"{cluster_label} ({info.get('thread_count', 0)} threads)")
                lines.append("-" * 80)
                for idx, thread in enumerate(info.get("threads", []), start=1):
                    subject = str(thread.get("subject") or "N/A").strip()
                    senders = str(thread.get("senders") or "N/A").strip()
                    last_date = str(thread.get("last_date") or "N/A").strip()
                    lines.append(f"{idx:>3}. {subject}")
                    lines.append(f"     From: {senders}")
                    lines.append(f"     Last activity: {last_date}")

                # If subclusters exist, include them in the summary
                subclusters = info.get("subclusters")
                if subclusters:
                    lines.append("")
                    lines.append("  Subclusters:")
                    for sub in subclusters:
                        lines.append(
                            f"    - {sub['label']} ({sub['thread_count']} threads)"
                        )
                        for sidx, thread in enumerate(sub.get("threads", [])[:5], start=1):
                            subject = str(thread.get("subject") or "N/A").strip()
                            senders = str(thread.get("senders") or "N/A").strip()
                            last_date = str(thread.get("last_date") or "N/A").strip()
                            lines.append(f"        {sidx:>3}. {subject}")
                            lines.append(f"             From: {senders}")
                            lines.append(f"             Last activity: {last_date}")

                lines.append("-" * 80)

        if noise_threads:
            lines.append("")
            lines.append("Noise / Outlier Threads")
            lines.append("-" * 80)
            for idx, thread in enumerate(noise_threads, start=1):
                subject = str(thread.get("subject") or "N/A").strip()
                senders = str(thread.get("senders") or "N/A").strip()
                lines.append(f"{idx:>3}. {subject} (From: {senders})")

        lines.append("")
        with open(output_path, "w", encoding="utf-8") as summary_file:
            summary_file.write("\n".join(lines))
        logger.info("Wrote cluster summary to %s", output_path)
    except OSError as exc:
        logger.error("Failed to write cluster summary to %s: %s", output_path, exc)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL_NAME = os.getenv(
    "OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"
)
LOCAL_EMBEDDING_MODEL_NAME = os.getenv(
    "LOCAL_EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large-instruct"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_GENERATION_MODEL = os.getenv(
    "GEMINI_GENERATION_MODEL", "gemini-2.5-flash"
)

USE_OPENAI_EMBEDDINGS = bool(OPENAI_API_KEY and OpenAI)
USE_GEMINI_FEATURES = bool(GEMINI_API_KEY and genai)

if USE_GEMINI_FEATURES:
    genai.configure(api_key=GEMINI_API_KEY)  # type: ignore[arg-type]

if USE_OPENAI_EMBEDDINGS:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)  # type: ignore[call-arg]
    EXPECTED_EMBEDDING_DIMENSION = 1536
else:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required when OPENAI_API_KEY is not set."
        )
    if torch is None:
        raise RuntimeError("PyTorch is required for local embeddings.")
    DEVICE = "cpu"
    if hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
        DEVICE = "cuda"
    elif (
        hasattr(torch, "backends")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):  # type: ignore[attr-defined]
        DEVICE = "mps"
    local_embedding_model = SentenceTransformer(  # type: ignore[call-arg]
        LOCAL_EMBEDDING_MODEL_NAME, device=DEVICE
    )
    EXPECTED_EMBEDDING_DIMENSION = (
        local_embedding_model.get_sentence_embedding_dimension()
    )

MODEL_MAX_INPUT_LENGTH_APPROX = 512
MAX_CONCAT_BODY_CHARS = 25000
CHARS_PER_CHUNK_TARGET = MODEL_MAX_INPUT_LENGTH_APPROX * 2
CHUNK_OVERLAP_CHARS = 150

# ---------------------------------------------------------------------------
# Text helpers & embeddings
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def clean_email_body_for_embedding_bs4(raw_body_data: Optional[str]) -> str:
    if not raw_body_data:
        return ""
    try:
        decoded_raw = base64.urlsafe_b64decode(raw_body_data)
        try:
            text = decoded_raw.decode("utf-8")
        except UnicodeDecodeError:
            text = decoded_raw.decode("latin-1", errors="replace")

        try:
            soup = BeautifulSoup(text, "lxml")
        except Exception:
            soup = BeautifulSoup(text, "html.parser")

        for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
            element.decompose()

        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[-_*=]{5,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 40 and any(
            phrase in text.lower()
            for phrase in ("unsubscribe", "view in browser", "sent from my")
        ):
            return ""
        return text
    except Exception as exc:  # pragma: no cover - parsing
        logger.error("Error cleaning email body: %s", exc)
        return ""


def find_text_plain(part: Optional[dict]) -> Optional[str]:
    if not part:
        return None

    mime_type = part.get("mimeType", "")
    body = part.get("body", {})
    body_data = body.get("data")

    if mime_type == "text/plain" and body_data and body.get("size", 0) > 0:
        return body_data
    if mime_type == "text/html" and body_data and body.get("size", 0) > 0:
        return body_data

    if mime_type.startswith("multipart/") and "parts" in part:
        html_fallback = None
        for subpart in part["parts"]:
            result = find_text_plain(subpart)
            if not result:
                continue
            sub_mime = subpart.get("mimeType")
            if sub_mime == "text/plain":
                return result
            if sub_mime == "text/html" and not html_fallback:
                html_fallback = result
        return html_fallback

    if part.get("filename") or body.get("attachmentId"):
        return None
    return None


def get_header(headers: Optional[Sequence[dict]], name: str) -> str:
    if not headers:
        return ""
    name_lower = name.lower()
    for header in headers:
        if header.get("name", "").lower() == name_lower:
            return header.get("value", "")
    return ""


def get_embedding_batch(texts: Sequence[str]) -> List[Optional[List[float]]]:
    if not texts:
        return []

    valid_texts = [text for text in texts if text and isinstance(text, str)]
    if not valid_texts:
        return [None] * len(texts)

    if USE_OPENAI_EMBEDDINGS and openai_client:
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL_NAME,
            input=valid_texts,
        )
        embedding_map = {
            text: record.embedding for text, record in zip(valid_texts, response.data)
        }
        return [embedding_map.get(text) for text in texts]

    with torch.no_grad():  # type: ignore[attr-defined]
        embeddings = local_embedding_model.encode(  # type: ignore[call-arg]
            valid_texts,
            convert_to_tensor=False,
            show_progress_bar=False,
            device=local_embedding_model.device,  # type: ignore[attr-defined]
        )
    if isinstance(embeddings, np.ndarray):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = embeddings
    embedding_map = {text: emb for text, emb in zip(valid_texts, embeddings_list)}
    return [embedding_map.get(text) for text in texts]


def make_gemini_model():
    if not USE_GEMINI_FEATURES:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY to enable labeling."
        )
    candidate_models: List[str] = []
    if GEMINI_GENERATION_MODEL:
        candidate_models.append(GEMINI_GENERATION_MODEL)
    candidate_models.extend(
        [
            "gemini-2.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-001",
            "gemini-1.0-pro",
        ]
    )

    last_error: Optional[Exception] = None
    for model_name in candidate_models:
        try:
            logger.debug("Attempting to initialize Gemini model '%s'", model_name)
            return genai.GenerativeModel(model_name)  # type: ignore[call-arg]
        except Exception as exc:  # pragma: no cover - network call
            last_error = exc
            logger.warning(
                "Gemini model '%s' unavailable (%s). Trying next fallback.",
                model_name,
                exc,
            )
            continue

    raise RuntimeError(
        f"Unable to initialize any Gemini model (tried: {candidate_models}). "
        f"Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------


def hash_email_for_id(email: Optional[str]) -> Optional[str]:
    if not email:
        return None
    return hashlib.sha256(email.encode("utf-8")).hexdigest()


def load_credentials(scopes: Sequence[str] = SCOPES) -> Credentials:
    creds = None
    if os.path.exists(GMAIL_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, scopes)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GMAIL_TOKEN_PATH, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        return creds

    if not os.path.exists(GMAIL_CLIENT_SECRET):
        raise FileNotFoundError(
            f"Gmail client secret file '{GMAIL_CLIENT_SECRET}' not found. "
            "Set GMAIL_CLIENT_SECRET env var or place credentials.json in the working directory."
        )

    flow = InstalledAppFlow.from_client_secrets_file(GMAIL_CLIENT_SECRET, scopes)
    creds = flow.run_local_server(port=0)
    with open(GMAIL_TOKEN_PATH, "w", encoding="utf-8") as token_file:
        token_file.write(creds.to_json())
    return creds


def get_gmail_service(user_email: Optional[str] = None):
    creds = load_credentials()
    service = build("gmail", "v1", credentials=creds)
    if user_email:
        try:
            profile = service.users().getProfile(userId="me").execute()
            email_address = profile.get("emailAddress")
            if email_address and email_address.lower() != user_email.lower():
                logger.warning(
                    "Authenticated Gmail account %s does not match supplied user email %s",
                    email_address,
                    user_email,
                )
        except HttpError as exc:  # pragma: no cover - network call
            logger.warning("Failed to fetch Gmail profile: %s", exc)
    return service


def create_label_if_missing(service, name: str, cache: Optional[dict] = None) -> Optional[str]:
    if cache is None:
        cache = {}
    if name in cache:
        return cache[name]

    try:
        labels_response = service.users().labels().list(userId="me").execute()
    except HttpError as exc:  # pragma: no cover - network call
        logger.error("Failed to list Gmail labels: %s", exc)
        return None

    for label in labels_response.get("labels", []):
        if label.get("name") == name:
            cache[name] = label.get("id")
            return cache[name]

    try:
        result = (
            service.users()
            .labels()
            .create(
                userId="me",
                body={
                    "name": name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            .execute()
        )
        cache[name] = result.get("id")
        return cache[name]
    except HttpError as exc:  # pragma: no cover - network call
        logger.error("Failed to create Gmail label %s: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


CHROMA_PERSIST_DIRECTORY = os.getenv(
    "CHROMA_PERSIST_DIRECTORY", os.path.join(os.getcwd(), "chroma_email_index")
)
CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "gmail_threads")
CHROMA_OVERSAMPLE_FACTOR = _positive_int_env("CHROMA_OVERSAMPLE_FACTOR", 6)
CHROMA_MAX_FETCH_LIMIT = _positive_int_env("CHROMA_MAX_FETCH_LIMIT", 8000)

_chroma_client = None
_chroma_lock = Lock()


def _embedding_model_suffix() -> str:
    if USE_OPENAI_EMBEDDINGS:
        short = OPENAI_EMBEDDING_MODEL_NAME.replace("/", "_")
        return "oai_emb3_small" if short == "text-embedding-3-small" else short
    short = LOCAL_EMBEDDING_MODEL_NAME.replace("/", "_")
    return "e5_large_multi" if short == "intfloat_multilingual-e5-large-instruct" else short


def _collection_name(sanitized_user_email: str) -> str:
    return f"{CHROMA_COLLECTION_PREFIX}_{_embedding_model_suffix()}_{sanitized_user_email}_v1"


def _ensure_chroma_client():
    if chromadb is None or CHROMA_IMPORT_ERROR:
        raise RuntimeError(
            "chromadb is not installed. Run `pip install chromadb` to continue."
        )
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    return _chroma_client


def _get_collection(sanitized_user_email: str):
    client = _ensure_chroma_client()
    name = _collection_name(sanitized_user_email)
    try:
        return client.get_collection(name=name)
    except Exception:  # pragma: no cover - chroma internal
        return client.get_or_create_collection(
            name=name,
            metadata={"source": "gmail_threads", "embedding_model": _embedding_model_suffix()},
        )


def _get_collection_by_name(collection_name: str):
    client = _ensure_chroma_client()
    return client.get_collection(name=collection_name)


def _count_documents(collection, sanitized_user_email: Optional[str]) -> int:
    try:
        if sanitized_user_email:
            try:
                return collection.count(where={"user_email": sanitized_user_email})
            except TypeError:
                # Older Chroma client versions do not accept a where filter for count().
                results = collection.get(
                    where={"user_email": sanitized_user_email},
                    include=[],
                )
                return len(results.get("ids", []))
        return collection.count()
    except Exception as exc:  # pragma: no cover - chroma internal
        logger.warning("Failed to count documents in Chroma collection: %s", exc)
        return 0


def _fetch_thread_vectors(
    collection,
    sanitized_user_email: Optional[str],
    max_threads: Optional[int],
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, Dict[str, Any]]]:
    limit = None
    if max_threads:
        limit = min(max_threads * CHROMA_OVERSAMPLE_FACTOR, CHROMA_MAX_FETCH_LIMIT)

    kwargs: Dict[str, Any] = {
        "include": ["embeddings", "metadatas"],
        "limit": limit,
    }
    if sanitized_user_email:
        kwargs["where"] = {"user_email": sanitized_user_email}

    results = collection.get(**kwargs)

    embeddings_raw = results.get("embeddings")
    if embeddings_raw is None:
        embeddings_list: List[Any] = []
    elif isinstance(embeddings_raw, np.ndarray):
        embeddings_list = embeddings_raw.tolist()
    else:
        embeddings_list = list(embeddings_raw)

    metadatas_raw = results.get("metadatas")
    if metadatas_raw is None:
        metadatas_list: List[Any] = []
    else:
        metadatas_list = list(metadatas_raw)

    if not embeddings_list or not metadatas_list:
        return [], {}, {}

    thread_vectors: Dict[str, List[float]] = {}
    thread_metadata: Dict[str, Dict[str, Any]] = {}
    thread_chunk_index: Dict[str, int] = {}
    thread_timestamp: Dict[str, int] = {}

    for embedding_raw, metadata in zip(embeddings_list, metadatas_list):
        if not metadata:
            continue
        thread_id = metadata.get("thread_id")
        if not thread_id:
            continue
        if embedding_raw is None:
            continue
        if isinstance(embedding_raw, np.ndarray):
            embedding = embedding_raw.tolist()
        else:
            embedding = list(embedding_raw)
        if not embedding:
            continue

        chunk_index = metadata.get("chunk_index", 0)
        timestamp = int(metadata.get("internal_date_ms") or 0)

        existing_index = thread_chunk_index.get(thread_id)
        existing_ts = thread_timestamp.get(thread_id, 0)

        update_record = False
        if existing_index is None:
            update_record = True
        elif chunk_index < existing_index:
            update_record = True
        elif chunk_index == existing_index and timestamp > existing_ts:
            update_record = True

        if not update_record:
            continue

        senders = metadata.get("senders") or "N/A"
        if isinstance(senders, list):
            senders = ", ".join(str(s) for s in senders[:5])
        subject = metadata.get("subject") or "N/A"
        snippet = metadata.get("snippet") or "N/A"
        last_date = metadata.get("last_date") or "N/A"
        message_count = metadata.get("message_count", 1)
        chunk_text = metadata.get("chunk_text") or ""

        thread_vectors[thread_id] = list(embedding)
        thread_metadata[thread_id] = {
            "thread_id": thread_id,
            "subject": str(subject),
            "senders": str(senders),
            "last_date": str(last_date),
            "message_count": message_count,
            "snippet": str(snippet),
            "chunk_text": str(chunk_text),
        }
        thread_chunk_index[thread_id] = chunk_index
        thread_timestamp[thread_id] = timestamp

    if not thread_vectors:
        return [], {}, {}

    ordered_ids = sorted(
        thread_vectors.keys(), key=lambda tid: thread_timestamp.get(tid, 0), reverse=True
    )
    if max_threads and len(ordered_ids) > max_threads:
        ordered_ids = ordered_ids[:max_threads]

    trimmed_vectors = {tid: thread_vectors[tid] for tid in ordered_ids}
    trimmed_metadata = {tid: thread_metadata[tid] for tid in ordered_ids}
    return ordered_ids, trimmed_vectors, trimmed_metadata


# ---------------------------------------------------------------------------
# Clustering logic & CLI
# ---------------------------------------------------------------------------


def cluster_emails(
    user_email: Optional[str] = None,
    collection_name: Optional[str] = None,
    min_cluster_size: int = 5,
    max_threads: int = 1000,
    apply_labels: bool = False,
    output_file: Optional[str] = None,
    enable_sub_labels: bool = False,
) -> Dict[str, Any]:
    if CHROMA_IMPORT_ERROR:
        raise RuntimeError(
            "ChromaDB dependency missing. Install chromadb to continue."
        ) from CHROMA_IMPORT_ERROR

    if not user_email and not collection_name:
        raise ValueError("Provide either user_email or collection_name for clustering.")

    if apply_labels and not user_email:
        raise ValueError("user_email is required when applying Gmail labels.")

    sanitized_email = hash_email_for_id(user_email) if user_email else None

    target_label = collection_name or user_email or "unknown"

    logger.info(
        "Starting clustering for %s (min_cluster_size=%d, max_threads=%d, apply_labels=%s, enable_sub_labels=%s)",
        target_label,
        min_cluster_size,
        max_threads,
        apply_labels,
        enable_sub_labels,
    )

    if collection_name:
        collection = _get_collection_by_name(collection_name)
    else:
        if sanitized_email is None:
            raise ValueError("Unable to derive collection without user_email.")
        collection = _get_collection(sanitized_email)

    collection_count = _count_documents(collection, sanitized_email)
    if collection_count == 0:
        raise RuntimeError(
            "No emails have been indexed yet. Run index.py first."
        )

    logger.info(
        "Chroma collection '%s' contains ~%d chunks",
        collection.name,
        collection_count,
    )

    ordered_ids, thread_vectors, thread_metadata = _fetch_thread_vectors(
        collection,
        sanitized_email,
        max_threads,
    )
    if not ordered_ids:
        raise RuntimeError("No emails found for clustering.")

    embeddings_array = np.vstack(
        [thread_vectors[thread_id] for thread_id in ordered_ids]
    ).astype(np.float32)
    if embeddings_array.size == 0:
        raise RuntimeError("No embeddings available for clustering.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(embeddings_array).tolist()

    clusters: Dict[int, List[Dict[str, Any]]] = {}
    noise_threads: List[Dict[str, Any]] = []
    total_assignments = len(cluster_labels)
    for idx, label in enumerate(cluster_labels, start=1):
        thread_id = ordered_ids[idx - 1]
        metadata = thread_metadata[thread_id]
        if label == -1:
            noise_threads.append(metadata)
        else:
            clusters.setdefault(label, []).append(metadata)

        logger.info(
            "Clustering progress: %d/%d threads processed (%d clusters, %d noise)",
            idx,
            total_assignments,
            len(clusters),
            len(noise_threads),
        )

    logger.info(
        "Clustering complete: %d clusters, %d noise threads",
        len(clusters),
        len(noise_threads),
    )

    gemini_model = None
    if USE_GEMINI_FEATURES:
        try:
            gemini_model = make_gemini_model()
        except Exception as exc:  # pragma: no cover - network call
            logger.warning("Gemini unavailable (%s). Cluster labels will be generic.", exc)

    labeled_clusters: Dict[str, Dict[str, Any]] = {}
    labelable_clusters = [
        (cluster_id, threads)
        for cluster_id, threads in clusters.items()
        if len(threads) >= 2
    ]
    total_label_targets = len(labelable_clusters)
    for label_index, (cluster_id, threads) in enumerate(labelable_clusters, start=1):
        cluster_label = f"Cluster {cluster_id}"
        if gemini_model:
            prompt = (
                "Analyze the following group of email threads and provide a concise, descriptive label for this cluster.\n\n"
                f"The cluster contains {len(threads)} email threads. Here are some examples:\n\n"
            )
            for idx, info in enumerate(threads[:5]):
                prompt += f"Thread {idx + 1}:\n"
                prompt += f"  Subject: {info['subject']}\n"
                prompt += f"  From: {info['senders']}\n"
                prompt += f"  Messages: {info['message_count']}\n\n"
            prompt += (
                "Based on these email threads, generate a short, descriptive label (2-4 words) that captures the main theme "
                "or topic of this cluster.\n\n"
                "Respond with ONLY the label, no explanation."
            )
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.3},  # type: ignore[arg-type]
                )
                if getattr(response, "text", None):
                    cluster_label = response.text.strip().replace('"', "")
            except Exception as exc:  # pragma: no cover - network call
                logger.warning("Gemini labeling failed for cluster %s: %s", cluster_id, exc)
                if "429" in str(exc):
                    time.sleep(2)  # mild backoff on rate limit

        labeled_clusters[cluster_label] = {
            "cluster_id": cluster_id,
            "label": cluster_label,
            "thread_count": len(threads),
            "threads": threads,
        }
        logger.info(
            "Cluster %s labeled as '%s' (%d threads)", cluster_id, cluster_label, len(threads)
        )
        logger.info(
            "Labeling progress: %d/%d clusters labeled",
            label_index,
            total_label_targets,
        )
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # Optional: sub-labeling (hierarchical clustering inside each cluster)
    # ------------------------------------------------------------------
    if enable_sub_labels and labeled_clusters and len(thread_vectors) > 0:
        logger.info("Sub-labeling enabled; performing hierarchical clustering inside clusters.")

        for parent_label, parent_info in labeled_clusters.items():
            parent_threads = parent_info.get("threads", [])
            if len(parent_threads) < max(4, min_cluster_size):
                # Too small to meaningfully sub-cluster
                continue

            # Collect embeddings for threads in this parent cluster
            sub_ids: List[str] = []
            sub_emb_list: List[List[float]] = []
            for t in parent_threads:
                tid = t.get("thread_id")
                if tid and tid in thread_vectors:
                    sub_ids.append(tid)
                    sub_emb_list.append(thread_vectors[tid])

            if len(sub_ids) < 4:
                continue

            sub_embeddings = np.vstack(sub_emb_list).astype(np.float32)

            # Use a smaller min_cluster_size for subclusters
            sub_min_cluster_size = max(2, min(len(sub_ids) // 2, max(3, min_cluster_size // 2)))
            logger.info(
                "Sub-clustering parent '%s' with %d threads (min_cluster_size=%d)",
                parent_label,
                len(sub_ids),
                sub_min_cluster_size,
            )

            try:
                sub_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=sub_min_cluster_size,
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                sub_labels = sub_clusterer.fit_predict(sub_embeddings).tolist()
            except Exception as exc:
                logger.warning(
                    "Sub-clustering failed for parent '%s': %s",
                    parent_label,
                    exc,
                )
                continue

            # Group threads by subcluster label
            subclusters_raw: Dict[int, List[Dict[str, Any]]] = {}
            for idx, sub_label in enumerate(sub_labels):
                if sub_label == -1:
                    continue  # treat as noise within the parent
                tid = sub_ids[idx]
                for t in parent_threads:
                    if t.get("thread_id") == tid:
                        subclusters_raw.setdefault(sub_label, []).append(t)
                        break

            if not subclusters_raw:
                continue

            # Label subclusters using Gemini, if available
            subcluster_list: List[Dict[str, Any]] = []
            for sub_id, sub_threads in subclusters_raw.items():
                sub_label_text = f"{parent_label} / Subcluster {sub_id}"
                if gemini_model:
                    sub_prompt = (
                        "You are labeling sub-groups of emails inside a larger category.\n"
                        f"The parent cluster label is: '{parent_label}'.\n"
                        f"This subcluster contains {len(sub_threads)} email threads. Here are some examples:\n\n"
                    )
                    for idx, info in enumerate(sub_threads[:5]):
                        sub_prompt += f"Thread {idx + 1}:\n"
                        sub_prompt += f"  Subject: {info['subject']}\n"
                        sub_prompt += f"  From: {info['senders']}\n"
                        sub_prompt += f"  Messages: {info['message_count']}\n\n"
                    sub_prompt += (
                        "Based on these emails, generate a short, descriptive sub-label (2-4 words)\n"
                        "that is MORE specific than the parent label.\n"
                        "Respond with ONLY the sub-label, no explanation."
                    )
                    try:
                        resp = gemini_model.generate_content(
                            sub_prompt,
                            generation_config={"temperature": 0.3},  # type: ignore[arg-type]
                        )
                        if getattr(resp, "text", None):
                            sub_label_text = resp.text.strip().replace('"', "")
                    except Exception as exc:
                        logger.warning(
                            "Gemini sub-labeling failed for parent '%s', subcluster %d: %s",
                            parent_label,
                            sub_id,
                            exc,
                        )

                subcluster_list.append(
                    {
                        "subcluster_id": sub_id,
                        "label": sub_label_text,
                        "thread_count": len(sub_threads),
                        "threads": sub_threads,
                    }
                )
                logger.info(
                    "Parent '%s': subcluster %d labeled as '%s' (%d threads)",
                    parent_label,
                    sub_id,
                    sub_label_text,
                    len(sub_threads),
                )

            if subcluster_list:
                parent_info["subclusters"] = subcluster_list

    label_cache: Dict[str, str] = {}
    labels_applied_successfully = False
    labeling_attempted = apply_labels
    total_labeling_time = 0.0
    all_requests: List[Any] = []


    if apply_labels and labeled_clusters:
        service = get_gmail_service(user_email)
        try:
            from collections import defaultdict

            # Map each thread_id -> set of Gmail label IDs to add
            thread_to_label_ids: Dict[str, set[str]] = defaultdict(set)

            total_label_clusters = len(labeled_clusters)
            for cluster_index, (cluster_label, cluster_info) in enumerate(
                    labeled_clusters.items(), start=1
            ):
                # Parent cluster label (e.g. "Offers and Points")
                parent_label_name = cluster_label
                parent_label_id = create_label_if_missing(
                    service, parent_label_name, cache=label_cache
                )
                if not parent_label_id:
                    logger.info(
                        "Label application skipped for cluster '%s' "
                        "(missing Gmail label id, %d/%d)",
                        cluster_label,
                        cluster_index,
                        total_label_clusters,
                    )
                    continue

                subclusters = cluster_info.get("subclusters") or []

                if subclusters:
                    # Nested labels for each subcluster
                    for sub in subclusters:
                        sub_short_label = (sub.get("label") or "").strip() or "Subcluster"
                        # Gmail uses "/" to make hierarchical labels
                        full_label_name = f"{parent_label_name}/{sub_short_label}"
                        sub_label_id = create_label_if_missing(
                            service, full_label_name, cache=label_cache
                        )
                        if not sub_label_id:
                            logger.warning(
                                "Could not create/find Gmail label for subcluster '%s'",
                                full_label_name,
                            )
                            continue

                        for thread in sub.get("threads", []):
                            tid = thread["thread_id"]
                            thread_to_label_ids[tid].add(parent_label_id)
                            thread_to_label_ids[tid].add(sub_label_id)

                    # Some parent threads might not end up in any subcluster:
                    subcluster_thread_ids = {
                        t["thread_id"]
                        for sub in subclusters
                        for t in sub.get("threads", [])
                    }
                    for thread in cluster_info["threads"]:
                        tid = thread["thread_id"]
                        if tid not in subcluster_thread_ids:
                            thread_to_label_ids[tid].add(parent_label_id)
                else:
                    # No subclusters – only the parent label
                    for thread in cluster_info["threads"]:
                        tid = thread["thread_id"]
                        thread_to_label_ids[tid].add(parent_label_id)

                logger.info(
                    "Label application prep: %d/%d clusters queued (%d threads accumulated)",
                    cluster_index,
                    total_label_clusters,
                    len(thread_to_label_ids),
                )

            # Turn the mapping into actual Gmail modify() requests
            for tid, label_ids in thread_to_label_ids.items():
                request = (
                    service.users()
                    .threads()
                    .modify(
                        userId="me",
                        id=tid,
                        body={"addLabelIds": list(label_ids)},
                    )
                )
                all_requests.append(request)

        except Exception as exc:  # pragma: no cover - network call
            logger.error("Error preparing Gmail label requests: %s", exc)
            all_requests = []

        if all_requests:
            labels_applied_successfully = True
            batch_size = 20
            max_execution_time = 25
            start_time = time.time()
            requests_processed = 0
            for start in range(0, len(all_requests), batch_size):
                elapsed = time.time() - start_time
                if elapsed > max_execution_time:
                    logger.warning(
                        "Timeout while applying labels (%d/%d requests processed)",
                        start,
                        len(all_requests),
                    )
                    labels_applied_successfully = False
                    break
                batch = all_requests[start : start + batch_size]
                batch_request = service.new_batch_http_request()
                for idx, req in enumerate(batch):
                    batch_request.add(req, request_id=f"cluster_{start}_{idx}")
                try:
                    batch_request.execute()
                    requests_processed += len(batch)
                    logger.info(
                        "Label application progress: %d/%d requests applied",
                        requests_processed,
                        len(all_requests),
                    )
                except Exception as exc:  # pragma: no cover - network call
                    logger.warning("Batch labeling error: %s", exc)
                    labels_applied_successfully = False
                if start + batch_size < len(all_requests):
                    time.sleep(0.5)
            total_labeling_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("EMAIL CLUSTERING RESULTS")
    print("=" * 80)
    print(f"Total threads processed: {len(thread_vectors)}")
    print(f"Clusters found: {len(labeled_clusters)}")
    print(f"Noise/outlier threads: {len(noise_threads)}")
    print()
    for cluster_label, cluster_info in labeled_clusters.items():
        print(f"CLUSTER: {cluster_label}")
        print(f"   Threads: {cluster_info['thread_count']}")
        print("   Sample subjects:")
        for thread in cluster_info["threads"][:3]:
            print(f"   - {thread['subject'][:60]}...")

        # Optional subclusters
        subclusters = cluster_info.get("subclusters")
        if subclusters:
            print("   Subclusters:")
            for sub in subclusters:
                print(f"      - {sub['label']} ({sub['thread_count']} threads)")
                for t in sub["threads"][:2]:
                    print(f"         * {t['subject'][:60]}...")
        print()

    if noise_threads:
        print("NOISE/OUTLIER THREADS:")
        for thread in noise_threads[:5]:
            print(f"   - {thread['subject'][:60]}...")
        print()
    print("=" * 80)

    if not apply_labels:
        message = "Email clustering completed. Gmail labeling skipped (pass --apply-labels to enable)."
    elif labels_applied_successfully:
        message = "Email clustering completed with Gmail labels applied."
    else:
        message = "Email clustering completed, but Gmail labeling encountered issues."

    if output_file:
        _write_cluster_summary(output_file, labeled_clusters, noise_threads)

    results = {
        "message": message,
        "results": {
            "total_threads_processed": len(thread_vectors),
            "clusters_found": len(labeled_clusters),
            "noise_threads_count": len(noise_threads),
            "total_requests_attempted": len(all_requests),
            "labeling_time_seconds": round(total_labeling_time, 2),
            "labeling_attempted": labeling_attempted,
            "labels_applied_successfully": labels_applied_successfully,
            "clusters": labeled_clusters,
            "noise_threads": noise_threads[:20] if noise_threads else [],
            "timestamp": datetime.now().isoformat(),
            "summary_output_file": output_file,
        },
    }
    return results


def _write_ics_events(events: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write a list of events to an .ics file.

    Event dict keys expected:
      - uid (str)
      - dtstart (str, in UTC like 20250101T090000Z)
      - dtend   (str, same format)
      - summary (str)
      - description (str)
      - location (str, optional)
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//gmail-tools//email-cluster-export//EN",
    ]
    for ev in events:
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{ev.get('uid')}",
                f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
                f"DTSTART:{ev.get('dtstart')}",
                f"DTEND:{ev.get('dtend')}",
                f"SUMMARY:{ev.get('summary','')}",
                f"DESCRIPTION:{ev.get('description','')}",
            ]
        )
        location = ev.get("location")
        if location:
            lines.append(f"LOCATION:{location}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines))
    logger.info("Wrote %d events to %s", len(events), output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster Gmail threads stored in ChromaDB.")
    parser.add_argument(
        "--user-email",
        help="Email address used during indexing (required if applying Gmail labels).",
    )
    parser.add_argument(
        "--collection",
        help="Explicit ChromaDB collection name to cluster. Overrides user-email namespace lookup.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Minimum number of items per cluster (default 5).",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=1000,
        help="Maximum number of thread embeddings to consider (default 1000).",
    )
    parser.add_argument(
        "--apply-labels",
        action="store_true",
        help="Apply generated labels back to Gmail threads.",
    )
    parser.add_argument(
        "--enable-sub-labels",
        action="store_true",
        help="Enable hierarchical sub-labeling within each top-level cluster.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to write a detailed text summary of clusters.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING). Default INFO.",
    )
    parser.add_argument(
        "--export-meetings-ics",
        help="Optional path to write meeting-like emails as an .ics calendar file.",
    )
    parser.add_argument(
        "--export-tasks",
        help="Optional path to write extracted tasks (Markdown/CSV) from emails.",
    )
    return parser.parse_args(argv)


def extract_meetings_from_clusters(
    clusters: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Very lightweight meeting extractor.

    For each thread, we send subject + snippet + chunk_text to Gemini
    and ask for a single event (or 'none') in JSON.
    """
    if not USE_GEMINI_FEATURES:
        logger.warning("Gemini not configured; meeting extraction will be skipped.")
        return []

    try:
        model = make_gemini_model()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to initialize Gemini for meeting extraction: %s", exc)
        return []

    events: List[Dict[str, Any]] = []
    for cluster_label, info in clusters.items():
        for thread in info.get("threads", []):
            text = (
                f"Subject: {thread.get('subject','')}\n"
                f"From: {thread.get('senders','')}\n"
                f"Snippet: {thread.get('snippet','')}\n"
                f"BodyExcerpt: {thread.get('chunk_text','')}\n"
            )
            prompt = (
                "You are an assistant that extracts meeting events from emails.\n"
                "Given the following email content, decide whether it describes a meeting or call.\n"
                "If yes, output a single JSON object with keys:\n"
                '  "title", "start_utc", "end_utc", "location", "notes"\n'
                "Use ISO-like UTC datetime format: YYYYMMDDTHHMMSSZ.\n"
                "If nothing looks like a scheduled meeting, respond with exactly: NONE\n\n"
                f"EMAIL:\n{text}"
            )
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1},  # type: ignore[arg-type]
                )
                raw = (resp.text or "").strip()
            except Exception as exc:  # pragma: no cover
                logger.warning("Gemini error while extracting meeting: %s", exc)
                continue

            if not raw or raw.upper().startswith("NONE"):
                continue

            # Very minimal JSON parsing with protection
            import json

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Could not parse meeting JSON: %s", raw)
                continue

            title = data.get("title") or thread.get("subject", "Meeting")
            start = data.get("start_utc")
            end = data.get("end_utc")
            if not start or not end:
                continue

            uid = f"{thread.get('thread_id')}-meeting"
            event = {
                "uid": uid,
                "dtstart": start,
                "dtend": end,
                "summary": title,
                "description": data.get("notes", "")[:500],
                "location": data.get("location", ""),
            }
            events.append(event)

    logger.info("Extracted %d potential meeting events", len(events))
    return events


def extract_tasks_from_clusters(
    clusters: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract action items from clustered emails using Gemini.

    Returns a list of tasks:
      - title
      - source_subject
      - source_senders
      - due (optional, free-text or YYYY-MM-DD)
      - priority (LOW/MEDIUM/HIGH)
    """
    if not USE_GEMINI_FEATURES:
        logger.warning("Gemini not configured; task extraction will be skipped.")
        return []

    try:
        model = make_gemini_model()
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to initialize Gemini for task extraction: %s", exc)
        return []

    tasks: List[Dict[str, Any]] = []

    for cluster_label, info in clusters.items():
        for thread in info.get("threads", []):
            text = (
                f"Subject: {thread.get('subject', '')}\n"
                f"From: {thread.get('senders', '')}\n"
                f"Snippet: {thread.get('snippet', '')}\n"
                f"BodyExcerpt: {thread.get('chunk_text', '')}\n"
            )
            prompt = (
                "You are an assistant that extracts TODO tasks from emails.\n"
                "From the following email, list clear action items as a JSON array.\n"
                "Each item should be an object with keys:\n"
                '  "title", "due", "priority"\n'
                'where "priority" is one of LOW, MEDIUM, HIGH, and "due" can be "" if unknown.\n'
                "If there are no tasks, respond with [] (empty JSON array).\n\n"
                f"EMAIL:\n{text}"
            )

            try:
                resp = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2},  # type: ignore[arg-type]
                )
                raw = (resp.text or "").strip()
            except Exception as exc:  # pragma: no cover
                logger.warning("Gemini error while extracting tasks: %s", exc)
                continue

            import json

            try:
                arr = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Could not parse task JSON: %s", raw)
                continue

            if not isinstance(arr, list):
                continue

            for item in arr:
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                tasks.append(
                    {
                        "title": title,
                        "due": (item.get("due") or "").strip(),
                        "priority": (item.get("priority") or "MEDIUM").upper(),
                        "source_subject": thread.get("subject", ""),
                        "source_senders": thread.get("senders", ""),
                    }
                )

    logger.info("Extracted %d tasks from email clusters", len(tasks))
    return tasks


def write_tasks_markdown(tasks: List[Dict[str, Any]], output_path: str) -> None:
    lines = ["# Email Tasks", ""]
    if not tasks:
        lines.append("_No tasks found._")
    else:
        for t in tasks:
            line = f"- [ ] **{t['title']}**"
            if t.get("due"):
                line += f" (due: {t['due']})"
            line += f" — _priority: {t['priority']}_"
            line += (
                f"\n  from: `{t['source_subject']}` ({t['source_senders']})"
            )
            lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %d tasks to %s", len(tasks), output_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    if not args.user_email and not args.collection:
        print("Provide --collection or --user-email to select a Chroma dataset.")
        return 1
    if args.apply_labels and not args.user_email:
        print("--apply-labels requires --user-email.")
        return 1
    try:
        result = cluster_emails(
            user_email=args.user_email,
            collection_name=args.collection,
            min_cluster_size=args.min_cluster_size,
            max_threads=args.max_threads,
            apply_labels=args.apply_labels,
            output_file=args.output_file,
            enable_sub_labels=args.enable_sub_labels,
        )
    except Exception as exc:
        logger.error("Clustering failed: %s", exc, exc_info=True)
        print(f"Clustering failed: {exc}")
        return 1

    clusters = result["results"]["clusters"]

    # export meetings as ICS if requested
    if args.export_meetings_ics:
        events = extract_meetings_from_clusters(clusters)
        _write_ics_events(events, args.export_meetings_ics)

    # export tasks if requested
    if args.export_tasks:
        tasks = extract_tasks_from_clusters(clusters)
        write_tasks_markdown(tasks, args.export_tasks)

    return 0


__all__ = [
    "CHARS_PER_CHUNK_TARGET",
    "CHUNK_OVERLAP_CHARS",
    "CHROMA_IMPORT_ERROR",
    "EXPECTED_EMBEDDING_DIMENSION",
    "MAX_CONCAT_BODY_CHARS",
    "USE_GEMINI_FEATURES",
    "chunk_text",
    "clean_email_body_for_embedding_bs4",
    "create_label_if_missing",
    "find_text_plain",
    "get_embedding_batch",
    "get_gmail_service",
    "get_header",
    "hash_email_for_id",
    "configure_logging",
    "logger",
    "_get_collection",
    "_get_collection_by_name",
    "cluster_emails",
]


if __name__ == "__main__":
    raise SystemExit(main())
