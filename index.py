# -*- coding: utf-8 -*-
"""Index Gmail threads into a local ChromaDB store."""

import argparse
import math
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

if sys.version_info < (3, 10):
    raise RuntimeError(
        "Python 3.10 or higher is required to run these Gmail tools. "
        f"Current interpreter: {sys.version.split()[0]}"
    )

from googleapiclient.errors import HttpError

from cluster import (
    CHARS_PER_CHUNK_TARGET,
    CHUNK_OVERLAP_CHARS,
    CHROMA_IMPORT_ERROR,
    EXPECTED_EMBEDDING_DIMENSION,
    MAX_CONCAT_BODY_CHARS,
    configure_logging,
    _get_collection,  # pylint: disable=protected-access
    chunk_text,
    clean_email_body_for_embedding_bs4,
    find_text_plain,
    get_embedding_batch,
    get_gmail_service,
    get_header,
    hash_email_for_id,
    logger,
)


def _collect_senders(messages: Iterable[Dict[str, Any]]) -> str:
    senders = set()
    for message in messages:
        headers = message.get("payload", {}).get("headers", [])
        sender_value = get_header(headers, "From")
        if not sender_value:
            continue
        if "<" in sender_value and ">" in sender_value:
            sender_value = sender_value.split("<", 1)[1].split(">", 1)[0]
        senders.add(sender_value.strip().lower())
    return ", ".join(sorted(senders))


def _extract_thread_body(messages: List[Dict[str, Any]]) -> str:
    combined_segments = []
    separator = "\n\n--- MESSAGE BREAK ---\n\n"
    total_chars = 0

    for message in messages:
        payload = message.get("payload", {})
        raw_body = find_text_plain(payload)
        cleaned_body = clean_email_body_for_embedding_bs4(raw_body) if raw_body else ""
        if not cleaned_body:
            continue

        if combined_segments:
            combined_segments.append(separator)
        combined_segments.append(cleaned_body)
        total_chars += len(cleaned_body)

        if total_chars >= MAX_CONCAT_BODY_CHARS:
            combined_segments[-1] = combined_segments[-1][:MAX_CONCAT_BODY_CHARS] + "..."
            break

    return "".join(combined_segments)


def _prepare_chunks_for_thread(
    thread_id: str,
    messages: List[Dict[str, Any]],
    snippet: str,
    sanitized_email: str,
) -> List[Dict[str, Any]]:
    if not messages:
        return []

    subject = get_header(messages[0].get("payload", {}).get("headers", []), "Subject")
    senders = _collect_senders(messages)
    message_count = len(messages)
    last_message = messages[-1]
    internal_date_ms = int(last_message.get("internalDate", 0))

    try:
        last_date_iso = datetime.utcfromtimestamp(internal_date_ms / 1000.0).isoformat() + "Z"
    except Exception:
        last_date_iso = ""

    body_text = _extract_thread_body(messages)
    if not body_text:
        return []

    chunks = chunk_text(body_text, CHARS_PER_CHUNK_TARGET, CHUNK_OVERLAP_CHARS)
    records: List[Dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            continue

        text_for_embedding = f"passage: Subject: {subject}\n\n{chunk}"
        metadata = {
            "thread_id": thread_id,
            "subject": subject[:250],
            "senders": senders[:200],
            "last_date": last_date_iso[:100],
            "message_count": message_count,
            "snippet": snippet[:160],
            "chunk_index": chunk_index,
            "total_chunks": len(chunks),
            "internal_date_ms": internal_date_ms,
            "user_email": sanitized_email,
            "chunk_text": chunk[:500],
        }

        records.append(
            {
                "id": f"{thread_id}_{chunk_index}",
                "text": text_for_embedding,
                "metadata": metadata,
            }
        )

    return records


def _fetch_thread_ids(service, max_threads: int, page_size: int) -> List[str]:
    thread_ids: List[str] = []
    page_token = None

    while len(thread_ids) < max_threads:
        request_size = min(page_size, max_threads - len(thread_ids))
        try:
            response = (
                service.users()
                .threads()
                .list(
                    userId="me",
                    maxResults=request_size,
                    pageToken=page_token,
                    includeSpamTrash=False,
                )
                .execute()
            )
        except HttpError as exc:
            logger.error("Error listing Gmail threads: %s", exc)
            break

        threads = response.get("threads", [])
        if not threads:
            break

        thread_ids.extend(thread.get("id") for thread in threads if "id" in thread)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

        time.sleep(0.3 + random.uniform(0, 0.2))

    return thread_ids[:max_threads]


RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
THREAD_FETCH_MAX_RETRIES = 4
THREAD_FETCH_BACKOFF_BASE = 0.75


def _retrieve_thread(service, thread_id: str) -> Dict[str, Any] | None:
    for attempt in range(1, THREAD_FETCH_MAX_RETRIES + 1):
        try:
            thread = (
                service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
            time.sleep(0.05)  # gentle pacing
            return thread
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status in RETRYABLE_HTTP_STATUSES and attempt < THREAD_FETCH_MAX_RETRIES:
                sleep_seconds = THREAD_FETCH_BACKOFF_BASE * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.3)
                logger.warning(
                    "Retryable error fetching thread %s (status %s, attempt %d/%d). Retrying in %.2fs",
                    thread_id,
                    status,
                    attempt,
                    THREAD_FETCH_MAX_RETRIES,
                    sleep_seconds + jitter,
                )
                time.sleep(sleep_seconds + jitter)
                continue

            logger.warning("Failed to fetch thread %s: %s", thread_id, exc)
            break
    return None


def index_gmail_threads(
    user_email: str,
    max_threads: int,
    page_size: int,
    embed_batch: int,
    reset_first: bool,
) -> Dict[str, Any]:
    if CHROMA_IMPORT_ERROR:
        raise RuntimeError(
            "chromadb is not installed. Install the chromadb package to enable local indexing."
        ) from CHROMA_IMPORT_ERROR

    logger.info(
        "Starting Gmail indexing for %s (max_threads=%d, page_size=%d, embed_batch=%d)",
        user_email,
        max_threads,
        page_size,
        embed_batch,
    )

    service = get_gmail_service(user_email)
    sanitized_email = hash_email_for_id(user_email)
    if not sanitized_email:
        raise ValueError("Valid user email is required.")

    collection = _get_collection(sanitized_email)

    if reset_first:
        logger.info("Clearing existing Chroma entries for %s", user_email)
        collection.delete(where={"user_email": sanitized_email})

    logger.info("Requesting thread ids from Gmail")
    thread_ids = _fetch_thread_ids(service, max_threads=max_threads, page_size=page_size)
    logger.info("Fetched %d thread ids to process", len(thread_ids))

    prepared_chunks: List[Dict[str, Any]] = []
    total_threads = 0
    total_chunks = 0

    total_thread_candidates = len(thread_ids)

    for idx, thread_id in enumerate(thread_ids, start=1):
        logger.debug("Fetching thread %s", thread_id)
        thread_data = _retrieve_thread(service, thread_id)
        if not thread_data:
            logger.info(
                "Thread progress: %d/%d candidates processed "
                "(%d indexed, %d chunks prepared)",
                idx,
                total_thread_candidates,
                total_threads,
                total_chunks,
            )
            continue

        messages = thread_data.get("messages", [])
        snippet = thread_data.get("snippet", "")
        chunk_records = _prepare_chunks_for_thread(
            thread_id=thread_id,
            messages=messages,
            snippet=snippet,
            sanitized_email=sanitized_email,
        )
        if not chunk_records:
            logger.debug("Thread %s produced no chunks; skipping", thread_id)
            continue

        prepared_chunks.extend(chunk_records)
        total_threads += 1
        total_chunks += len(chunk_records)
        logger.debug(
            "Thread %s contributed %d chunks (total_chunks=%d)",
            thread_id,
            len(chunk_records),
            total_chunks,
        )

        logger.info(
            "Thread progress: %d/%d candidates processed "
            "(%d indexed, %d chunks prepared)",
            idx,
            total_thread_candidates,
            total_threads,
            total_chunks,
        )

    logger.info(
        "Prepared %d chunks across %d threads. Embedding in batches of %d",
        total_chunks,
        total_threads,
        embed_batch,
    )

    total_batches = max(1, math.ceil(len(prepared_chunks) / embed_batch)) if prepared_chunks else 0
    embedded_chunks = 0

    for batch_index, start in enumerate(range(0, len(prepared_chunks), embed_batch), start=1):
        batch_records = prepared_chunks[start : start + embed_batch]
        logger.debug(
            "Embedding batch starting at index %d (%d records)", start, len(batch_records)
        )
        texts = [record["text"] for record in batch_records]
        embeddings = get_embedding_batch(texts)

        valid_ids = []
        valid_embeddings = []
        valid_metadatas = []

        for record, embedding in zip(batch_records, embeddings):
            if not embedding:
                logger.warning("Skipping chunk %s with missing embedding", record["id"])
                continue
            if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
                logger.warning(
                    "Skipping chunk %s due to embedding dim mismatch (%s)",
                    record["id"],
                    len(embedding),
                )
                continue
            valid_ids.append(record["id"])
            valid_embeddings.append(list(embedding))
            valid_metadatas.append(record["metadata"])

        if not valid_ids:
            continue

        collection.add(
            ids=valid_ids,
            embeddings=valid_embeddings,
            metadatas=valid_metadatas,
        )
        logger.debug("Persisted %d embeddings to Chroma", len(valid_ids))
        embedded_chunks += len(valid_ids)
        logger.info(
            "Embedding progress: batch %d/%d complete (%d/%d chunks stored)",
            batch_index,
            total_batches,
            embedded_chunks,
            total_chunks,
        )

    logger.info(
        "Indexing complete for %s: %d threads processed, %d chunks stored",
        user_email,
        total_threads,
        total_chunks,
    )
    return {
        "threads_processed": total_threads,
        "chunks_written": total_chunks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index Gmail threads into ChromaDB.")
    parser.add_argument("--user-email", required=True, help="Email to index.")
    parser.add_argument(
        "--max-threads",
        type=int,
        default=100,
        help="Maximum number of threads to pull (default 100).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Threads per page when listing Gmail threads (default 100).",
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=64,
        help="Number of chunks to embed per batch (default 64).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing Chroma entries for this user before indexing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING). Default INFO.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    logger.info("CLI arguments parsed successfully")
    try:
        result = index_gmail_threads(
            user_email=args.user_email,
            max_threads=args.max_threads,
            page_size=args.page_size,
            embed_batch=args.embed_batch,
            reset_first=args.reset,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Indexing failed: %s", exc, exc_info=True)
        print(f"Indexing failed: {exc}")
        return 1

    print(
        f"Indexed {result['chunks_written']} chunks across "
        f"{result['threads_processed']} threads for {args.user_email}"
    )
    logger.info(
        "Successfully indexed %d chunks across %d threads for %s",
        result["chunks_written"],
        result["threads_processed"],
        args.user_email,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
