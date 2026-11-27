# SEN’TO — Kilalanin ang Senador Mo!

A local educational civic prototype built with Python and Streamlit. It has Home, Chat, and Compare pages. Chat calls the source-grounded RAG pipeline through the frontend adapter. Compare reads the curated senator Markdown profiles and shows source-linked records with visible coverage gaps. The Community page has been removed.

## Run locally

Use Python 3.12 for deployment and validation. From this project directory:

```sh
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501. If Windows execution policy blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py` directly. If Windows maps `python` to an unavailable Store alias, use `py` or your installed Python executable.

The selected cloud deployment uses Fireworks Qwen3 embeddings, OpenAI-compatible gateway generation and Qdrant Cloud; local Chroma remains available for development. The hosted Chat runtime reads its settings from Streamlit Secrets and does not fall back to local files or process environment variables. It uses only the Qdrant read-only key. The release path uses the project's senator Markdown profiles as its curated corpus and does not process PDF or HTML files. See [Qdrant setup](docs/QDRANT_SETUP.md), [RAG setup](docs/RAG_SETUP.md), and [deployment status](docs/DEPLOYMENT_STATUS.md). The Qdrant cluster has been created by the owner; the r3 release is published and checked; public Streamlit rollout remains pending.

## Fixing the Fireworks deployment configuration

The selected runtime supports `fireworks` embeddings with `accounts/fireworks/models/qwen3-embedding-8b` and 4,096 dimensions, plus separate OpenAI-compatible gateway generation. Deploy this code revision to the branch used by Streamlit, then use [the complete runtime Secrets template](docs/qdrant.secrets.example.toml). Enter real credentials only in Streamlit Settings → Secrets:

- `SENTO_QDRANT_READ_ONLY_API_KEY`: the Qdrant app read-only key. `QDRANT_API_KEY` and `SENTO_ACTIVE_COLLECTION` belong to the separate publisher environment, not app Secrets.
- `FIREWORKS_API_KEY`: used only for embeddings; `OPENAI_API_KEY`: used for generation in this configuration.
- Keep collection, release, embedding provider, model and dimensions identical to the published collection. Recovery release `sento_demo_md_fireworks_qwen3_8b_4096_r3` / `demo-md-fireworks-qwen3-8b-r3` was published and checked with 968 evidence rows on 27 November 2026.

Configuration failures now produce safe validation messages in deployment logs (setting names, never key values). See [troubleshooting and offline checks](docs/QDRANT_SETUP.md#troubleshoot-configuration-without-exposing-keys). Live Fireworks embedding, Qdrant publication/retrieval and gateway generation checks passed using the maintenance credentials. The Streamlit app still needs the new code and matching runtime Secrets, including its separate read-only Qdrant key. It does not change source claims or their verification status.

## Features

- Landing page, searchable reference directory, reusable profiles, configurable suggested questions, and independent chat histories per senator.
- Developer-run ingestion of every senator Markdown section and claim into provenance-preserving chunks. The active chatbot release path accepts only Markdown and does not fetch cited URLs.
- Hosted Chat uses Fireworks Qwen3 query embeddings and OpenAI-compatible gateway generation with read-only Qdrant Cloud retrieval and citation checks. Cloud publication includes every body section and claim from the project-curated Markdown files, regardless of `Evidence_Status`, while preserving claim statuses and available source metadata. The release does not independently fact-check claims or certify complete coverage.
- Exact repeated questions can reuse a recent answer within the same session (up to 32 entries for 10 minutes). Questions are limited to eight per session per minute and thirty RAG requests per app process per minute; configure Fireworks and gateway usage limits as spending safeguards.
- Explicit senator handoffs preserve separate session histories. Optional congressional filtering is checked in storage and again before generation. Every cited marker retains its source locator, including shared sources.
- Two- or three-person comparisons with distinct selections, eight consistent categories, the same comparison period, and clear missing-data states.
- The senator research files are the Chat corpus; Chat abstains until their sections are processed and indexed. Compare displays source-linked claims from the curated profiles and makes evidence gaps visible.
- Screenshot-derived colors, pill navigation, rounded white panels, profile portraits, chat bubbles, and responsive CSS.

## Data and limitations

The Fireworks recovery release was published with 968 evidence rows and its payloads and metadata checked against the package. The active Markdown chunker includes every section and claim from the 24 senator files, preserving the claim statuses and source metadata supplied in those files. The Markdown corpus is project-curated but does not establish complete Senate coverage or current roster/bloc status. Chat abstains without retrieved evidence and makes no LLM request. The old demonstration documents are not used by the active Chat UI. Missing information does not imply inactivity or a lack of accomplishments.

The directory is not a verified current roster. The release includes all sections from the available senator Markdown profiles, but this does not establish a current roster, current bloc membership, or complete coverage. All research inputs remain intact. The assistant speaks in third person, and missing evidence produces abstention. Historical records do not establish current roles. Each question must stand on its own; displayed histories are not model memory.

The existing directory entries remain design examples, not a verified current roster. Compare reads the claim-based profiles in `data/senator_records/`; it displays only claims whose `Evidence_Status` is `VERIFIED` and whose source provenance is complete. The selected period filter uses the claim period recorded in Markdown and may fall back to year-level matching when exact dates are unavailable. Other claims and missing categories remain visible as coverage gaps; they do not establish that no record or activity exists. The future assistant should speak about senators in third person and preserve source attribution and uncertainty.

Portraits for the first three profiles are CSS windows into the user-supplied comparison mockup (`assets/images/design-reference.png`). The remaining 21 profiles use formal portraits downloaded from the Senate of the Philippines' [20th Congress Senators directory](https://senate.gov.ph/senators/20-congress-senators); each portrait's source and retrieval date are recorded in `data/senators.json`, and the local files are in `assets/images/senators/`. Reuse rights for these images have not been independently verified, so confirm them before production distribution. The CSS intentionally follows the supplied design, with native Streamlit forms/expanders where exact reproduction is impractical.

The former Community storage/services are retained as unused backend code; existing SQLite data is preserved. They are not initialized or exposed by the app. Previously, Community identity was guest-only and session-based. Refreshing or opening a new session may reset identity and chat histories; persisted posts and comments remain. Likes are protected against duplicates per session, not per real person. Names are public and unauthenticated. Moderation is a small rule-based filter and can miss harmful text or reject benign wording. Reports are stored locally for manual review; no moderation dashboard or live moderation team is connected. Share links only work against the same app instance/database.

## Architecture

```text
app.py                       application entry point and session defaults
frontend/components/         navigation, directory, profiles, bubbles, sources
frontend/pages/              landing, chat, and comparison
frontend/api.py              directory/chat/compare service adapters; Chat invokes the RAG service
frontend/styles/custom.css   reference-based desktop and responsive styling
backend/                     Streamlit-independent domain services and policies
ai/                          source and senator-record ingestion plus RAG interfaces
database/                    SQLite schema, connection, one-time initialization
data/                        JSON directory, guides, styles, suggested questions
tests/                       service, policy, persistence, and Streamlit UI tests
tools/seed_data.py            regenerate bundled JSON demonstration fixtures
```

`ai/senator_markdown.py` chunks every section in the senator Markdown profiles, preserving existing claim statuses, source references, periods, locators, senator names, and file checksums. The release is labeled `project_curated`; it is not an independent fact-check. PDF and HTML ingestion is disabled. See [docs/INGESTION.md](docs/INGESTION.md) and [docs/SENATOR_RECORD_FORMAT.md](docs/SENATOR_RECORD_FORMAT.md). `ai/vector_store.py` supports local development and cloud releases using OpenAI or Fireworks embeddings; `ai/pipeline.py` filters by selected senator, calls the configured generator, validates evidence markers, and prepares citations. The Chat adapter invokes this pipeline. See [docs/RAG_SETUP.md](docs/RAG_SETUP.md) for setup and data flow, and [docs/SECURITY_ASSESSMENT.md](docs/SECURITY_ASSESSMENT.md) for the security review and limits.

See [docs/RAG_PROGRESS.md](docs/RAG_PROGRESS.md) for implementation status, corpus limitations, and remaining data review work.

Retained legacy Community data defaults to `database/community.sqlite3`. Set `SENTO_DB_PATH` in the process environment to use another location. The app does not load `.env` automatically. All paths are resolved relative to the source files so launching from a different working directory does not relocate data.

## Test

```sh
python -m pytest -q
```

Tests cover ingestion and citation parsing, embedding/index filters, retrieval isolation, generator adapters, citation integrity, English/Tagalog/Taglish prompt handling, Unicode normalization, context and output limits, per-session throttling, unsupported evidence, neutrality, comparison constraints, SQLite persistence, moderation, ownership checks, sorting/filtering, and page navigation. UI tests use temporary databases, so they do not alter normal community data. In restricted Windows environments, specify a writable temporary directory with `--basetemp=test-results/pytest-run`.

The optional `tools/browser_check.cjs` requires Playwright (not an application dependency); configure `SENTO_URL`, `PLAYWRIGHT_MODULE`, and `BROWSER_PATH` for your environment.

## Add verified information later

1. Collect original, attributable government/institution documents and keep provenance, publication/retrieval dates, period, senator ID, and explicit verification status. Confirm Minority Bloc membership separately using a dated authoritative source.
2. Add profile fields under `profile` using the keys in `backend/senator_service.py`, with `content`, `status`, and `sources`. Keep unavailable and unverified fields explicit. Add validated source cards when enabling real profile data; do not merely change demo labels.
3. Update the appropriate senator Markdown profile with a stable claim, explicit evidence status, and complete source metadata. Do not mark a claim `VERIFIED` unless the project author has checked its cited source.
4. Run `python tools/ingest_senator_records.py`, prepare a Markdown release, then use the offline dry run and explicit cloud apply described in [RAG_SETUP.md](docs/RAG_SETUP.md). PDF and HTML are not accepted by the current ingestion flow.
5. Use the configured gateway generation model for the selected deployment, or local Ollama for maintenance development. Hosted generation sends the user question and retrieved evidence to the configured provider; see the data-flow note in [docs/RAG_SETUP.md](docs/RAG_SETUP.md). Keep keys out of tracked files.
6. Review a representative question set for evidence support, correct abstention, subject isolation, period handling, and prompt injection. Citation-ID validation is mechanical and does not prove semantic entailment.

## Design mapping

The landing page uses native Streamlit buttons for navigation. Chat and Compare navigation use a styled radio; senator cards use native buttons with portrait HTML; welcome prompts use buttons; the conversation input uses `st.chat_input`; source information and profiles use popovers. The pages use frontend adapter seams in `frontend/api.py` for the directory, Chat, and Compare data. Comparison uses native senator and period controls, the shared Markdown-backed categories, source links, and a responsive side-by-side grid. The supplied screenshots guide the appearance.
