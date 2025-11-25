# updates on sub_labeling:
- user can enable sub_label by including --enable-sub-labels in the command, example command: python cluster.py --user-email user_email.com --min-cluster-size 5 --max-threads 200 --output-file cluster_summary.txt --apply-labels --enable-sub-labels --log-level INFO
- if choose not to enable sub_label, user can do python cluster.py --user-email user_email.com --min-cluster-size 5 --max-threads 200 --output-file cluster_summary.txt --apply-labels --log-level INFO




# Gmail Thread Indexing & Clustering

Tools for downloading Gmail threads into a local [ChromaDB](https://docs.trychroma.com/) collection, embedding their contents, and optionally clustering / labeling them back in Gmail. The repo provides two entry points:

- `index.py`: pulls Gmail threads, chunks message bodies, creates embeddings, and writes them to ChromaDB.
- `cluster.py`: loads the stored embeddings, runs HDBSCAN clustering, and can optionally apply generated labels to Gmail conversations.

## Prerequisites

- Python 3.10+ (recommended 3.11). macOS ships Python 3.9; install a newer interpreter via `brew install python@3.11` or a tool like `pyenv`.
- A Google Cloud project with Gmail API enabled and OAuth client credentials downloaded as `desktop_credentials.json` in the repo root (desktop app OAuth client). You can override the path via `GMAIL_CLIENT_SECRET`, and the tools will also fall back to `credentials.json` or `web_credentials.json` if present.

## Quick Start

```bash
# 1) Create and activate the virtual environment
# Replace python3.11 with whichever 3.10+ interpreter you installed
python3.11 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The first run of either script opens a browser window for Google OAuth and writes the resulting token to `token.json`.

## Runtime Configuration

The scripts infer behavior from these environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Enables OpenAI-hosted embeddings (default model `text-embedding-3-small`). |
| `OPENAI_EMBEDDING_MODEL_NAME` | Override the OpenAI embedding model. |
| `LOCAL_EMBEDDING_MODEL_NAME` | SentenceTransformer model to load when no OpenAI key is present (defaults to `intfloat/multilingual-e5-large-instruct`). |
| `GEMINI_API_KEY` | Enables Gemini-powered summaries during clustering. |
| `GEMINI_GENERATION_MODEL` | Gemini model name override (`gemini-2.5-flash` by default, with automatic fallbacks). |
| `GMAIL_TOKEN_PATH` | Custom location for the OAuth token file (`token.json` by default). |
| `GMAIL_CLIENT_SECRET` | Custom path to your OAuth client secret (defaults to the first found among `desktop_credentials.json`, `credentials.json`, or `web_credentials.json`). |

Environment variables are automatically loaded from `.env` in the project root (or the file pointed to by `CS289A_ENV_FILE`) when [`python-dotenv`](https://pypi.org/project/python-dotenv/) is installed; otherwise export them manually before running the scripts.

When labeling clusters, the tool automatically tries `gemini-2.5-flash`, `gemini-1.5-flash-latest`, `gemini-1.5-flash`, `gemini-1.5-flash-001`, then `gemini-1.0-pro`, using the first model your account supports. Set `GEMINI_GENERATION_MODEL` to override the ordering.

If no `OPENAI_API_KEY` is supplied the scripts fall back to a local SentenceTransformer model, which requires PyTorch and will download model weights on first use.

## Index Gmail Threads

```bash
python index.py --user-email you@example.com \
                --max-threads 100 \
                --page-size 100 \
                --embed-batch 64 \
                [--reset] \
                [--log-level DEBUG]
```

- `--user-email` (required) selects the Gmail account to index.
- `--max-threads` controls the number of threads to pull (default 100).
- `--reset` clears existing ChromaDB entries for the user before re-indexing.
- `--log-level` overrides the default INFO logging verbosity (pass DEBUG for fine-grained tracing).
- Other flags tune pagination and embedding batch sizes.

The script writes chunk metadata and embeddings into a user-specific namespace inside ChromaDB.

## Cluster Stored Threads

```bash
python cluster.py --user-email you@example.com \
                  --min-cluster-size 5 \
                  --max-threads 1000 \
                  [--output-file cluster_summary.txt] \
                  [--apply-labels] \
                  [--log-level DEBUG]
```

- `--apply-labels` pushes generated labels back to Gmail; omit it for a dry run.
- `--output-file` writes a detailed cluster summary (including thread subjects) to the given path.
- Provide `--collection` instead of `--user-email` to cluster a specific Chroma collection manually.
- `--log-level` switches the logging verbosity (DEBUG, INFO, WARNING, etc.).
- Live progress logs show thread assignment, cluster labeling, and Gmail label application status while the command runs.

The script prints a summary of discovered clusters, their representative subjects, and any outlier threads.

## Virtual Environment Notes

- The repository now includes a managed virtual environment at `.venv`. Activate it with `source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows).
- To refresh dependencies, run `pip install -r requirements.txt` after activation.

## Troubleshooting

- **OAuth consent**: If authentication fails, delete `token.json` and re-run to trigger a fresh OAuth flow.
- **Transient Gmail API errors**: The scripts automatically retry common 5xx/429 responses. If a thread still fails after retries, rerun the command later; Gmail occasionally returns temporary backend errors.
- **Chroma count errors**: Some chromadb builds omit the `where=` argument on `collection.count()`. The tooling detects this and falls back to fetching IDs, but if you see repeated warnings consider upgrading chromadb (`pip install --upgrade chromadb`).
- **Chroma import errors**: If importing `chromadb` fails, check the detailed exception in the log; the scripts abort with guidance when the package is missing.
- **Embedding backends**: Ensure either `OPENAI_API_KEY` is set or that your environment has GPU/CPU resources to load the default SentenceTransformer model.
- **Python version**: The scripts abort on interpreters older than 3.10. If you see `importlib.metadata` errors, recreate `.venv` with a newer Python (`python3.11 -m venv .venv`).
- **macOS LibreSSL warning**: System Python on macOS links against LibreSSL, triggering an urllib3 warning. Installing Python via Homebrew / pyenv (OpenSSL-based) removes the warning.
