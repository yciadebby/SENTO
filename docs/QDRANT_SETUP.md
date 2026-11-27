# Qdrant deployment for SEN’TO

Qdrant replaces Chroma Cloud for the selected deployment. Local Chroma development and the Chroma Cloud adapter remain available. The user supplied a healthy `sento-demo` cluster and active `sento-publisher` (manage) and `sento-app` (read) keys in a dashboard screenshot. This is account-setup evidence, not an authenticated application check.

Endpoint: `https://d9b16d6d-d989-4240-9c8c-cecffb5f36e1.ap-northeast-2-0.aws.cloud.qdrant.io`

## Runtime settings in Streamlit Cloud

The hosted chatbot reads its RAG configuration directly from root-level values in the deployed app's Streamlit Settings → Secrets. It does not read `.env`, local TOML files, or process environment variables for chatbot settings. Use [the settings template](qdrant.secrets.example.toml) as a checklist, not as a runtime file. Paste the non-secret settings there and enter credentials directly in Streamlit Secrets.

Set `SENTO_QDRANT_READ_ONLY_API_KEY` to the full `sento-app` read-only key, `FIREWORKS_API_KEY` to the Fireworks key, and `OPENAI_API_KEY` to the gateway-issued generation key. Do not put the `sento-publisher` manage key or `QDRANT_API_KEY` in app Secrets. The app passes only the separately named read-only secret to the Qdrant client; confirm the key's Read Cluster role in the Qdrant dashboard. Keep the publisher key in the maintainer's separate publishing environment.

Leave `SENTO_QDRANT_COLLECTION` and `SENTO_RELEASE_ID` unset until a project-curated Markdown release has been published and checked. Missing cloud settings fail closed; the chat runtime cannot switch to local Chroma or Ollama.

The owner previously reported a successful connection-only read using the app key. That verified authentication at that time, not a published collection, write restrictions, or OpenAI model access.

## Prepare Markdown evidence for publication

For the selected Fireworks release, use `SENTO_EMBEDDING_PROVIDER=fireworks`, model `accounts/fireworks/models/qwen3-embedding-8b`, 4096 dimensions and `https://api.fireworks.ai/inference/v1`. The separate Fireworks key is required for embeddings; the gateway-issued key remains in OPENAI_API_KEY for generation. [Fireworks documents the compatible embeddings API](https://docs.fireworks.ai/guides/querying-embeddings-models) and [the model path](https://fireworks.ai/models/fireworks/qwen3-embedding-8b); [Qwen documents its 4096-dimensional output](https://github.com/QwenLM/Qwen3-Embedding). Existing OpenAI releases remain supported, but their model/dimension settings must be restored together with their collection/release.

The chatbot release uses only `data/senator_records/*.md` and `*.markdown`; PDF and HTML ingestion is disabled. No separate review register is needed. It includes every body section and claim, regardless of `Evidence_Status`, preserves the status and source metadata written in each file, and records the source-file checksum. Follow [RAG_SETUP.md](RAG_SETUP.md), then run:

```powershell
.\.venv\Scripts\python.exe tools/ingest_senator_records.py
.\.venv\Scripts\python.exe tools/prepare_markdown_release.py --embedding-provider fireworks --release-id demo-md-fireworks-qwen3-8b-r3 --collection sento_demo_md_fireworks_qwen3_8b_4096_r3 --output-dir data/releases/demo-md-fireworks-qwen3-8b-r3
```

The checked r3 package already exists under `data/releases/demo-md-fireworks-qwen3-8b-r3/` and is published. Do not rerun publication against r3; use a new release ID, collection name and output directory for later revisions. The empty `data/releases/demo-r1/review.json` is historical and is not read by this workflow. No source file is edited or promoted by the exporter.

## Publish using the maintenance key

For the maintainer workflow, create a separate ignored publisher settings file. Enter the `sento-publisher` manage key as `QDRANT_API_KEY` and separate Fireworks/OpenAI credentials as `FIREWORKS_API_KEY` and `OPENAI_API_KEY`. Add `SENTO_ACTIVE_COLLECTION = "NONE"` for the first publication; for future releases use the actual serving collection name. Set the candidate collection/release to the manifest's matching values. Keep this publisher file out of Streamlit Secrets; the app uses only `SENTO_QDRANT_READ_ONLY_API_KEY`.

```powershell
.\.venv\Scripts\python.exe tools/index_corpus.py --secrets-file .streamlit/secrets.publisher.toml --chunks data/releases/demo-md-fireworks-qwen3-8b-r3/chunks.jsonl --release-manifest data/releases/demo-md-fireworks-qwen3-8b-r3/manifest.json --dry-run
.\.venv\Scripts\python.exe tools/index_corpus.py --secrets-file .streamlit/secrets.publisher.toml --chunks data/releases/demo-md-fireworks-qwen3-8b-r3/chunks.jsonl --release-manifest data/releases/demo-md-fireworks-qwen3-8b-r3/manifest.json --apply
.\.venv\Scripts\python.exe tools/check_rag.py --runtime-settings --secrets-file .streamlit/secrets.qdrant.toml --chunks data/releases/demo-md-fireworks-qwen3-8b-r3/chunks.jsonl --release-manifest data/releases/demo-md-fireworks-qwen3-8b-r3/manifest.json
```

Dry run is offline and does not require model or database credentials. Apply sends Markdown text to the configured embedding provider (Fireworks for the selected release) for paid embeddings and uploads vectors and evidence to Qdrant. It creates a new collection only; an existing name fails. Never use `recreate_collection` or replace the active release. Failed uploads stay inactive; retry with a new candidate collection and matching package. The full health check compares saved IDs and complete payloads against the package, with no model calls.

## Storage and retrieval contract

- One cosine collection per release. The selected Fireworks release uses 4,096-dimensional `accounts/fireworks/models/qwen3-embedding-8b` vectors. Existing OpenAI releases continue to use 1,536-dimensional `text-embedding-3-small` vectors. Qdrant inference is not used.
- Deterministic UUIDs map the existing `chunk_id::senator_id` row IDs to Qdrant points. Original IDs remain in each payload, along with complete text, citation sources, raw periods and normalized Congress.
- Keyword indexes are created before upload for `record_kind`, `senator_id`, `curation_status`, `congress_id`, `relevant_period`, and `release_id`. Original claim evidence statuses are preserved in payloads and are not used as an inclusion filter.
- One reserved control point stores release/model/schema identity, curation status, counts and `building`/`ready` state. It is excluded from every evidence query and from displayed coverage counts. The dashboard's total point count is therefore **expected evidence rows plus one**.
- Runtime opens existing collections and validates identity, dimensions, cosine metric, ready state and total count. It has no write methods. Every query filters evidence kind, project curation, release and selected senator, plus the requested period. Returned payloads are checked again; the pipeline independently checks subject and Congress.
- Qdrant cosine similarity is converted to SEN’TO's distance as `1 - similarity`; the existing maximum distance of 0.65 corresponds to a minimum similarity of 0.35. Evaluate the cutoff against reviewed questions before release.
- Provider connections are cached without histories. Exact repeat answers are cached only in the current session for ten minutes, up to 32 entries. The UI allows eight submissions per session/minute; cloud retrieval allows 30 requests per running app process/minute. These process/session controls do not replace Fireworks and gateway spending limits and alerts. Read/write requests are bounded by a 45-second client timeout. Maintenance scans use pagination; uploads use batches of 32. Existing conservative application payload limits remain in force; these are not advertised as Qdrant service quotas.

## Deploy only after live review

The configured model passed a gateway smoke test on 27 November 2026; recheck account access if the key, model or endpoint changes. Gateway requests use Chat Completions; the Responses reasoning option applies only to direct OpenAI requests. Complete the supported/unsupported question set and sentence-support review. Then deploy the code to the branch watched by the existing Streamlit app, using `app.py` and Python 3.12. Enter the runtime template's values in Streamlit Settings → Secrets using the **read-only** key and the published collection/release. Do not enter the publisher key. The app sends the current question and retrieved evidence to the owner-supplied OpenAI-compatible gateway for generation; query embeddings use Fireworks for the selected release, while vectors and source payloads live in Qdrant Cloud.

Test another device with the laptop off, cloud restart, and sleep/wake behavior. Keep the previous release and matching settings for rollback. Free Qdrant clusters can suspend after one week without activity and be deleted after four weeks without reactivation, so retain the versioned reviewed package for recovery. No account signup or current cloud retention promise replaces those acceptance checks.

References: [Qdrant authentication](https://qdrant.tech/documentation/cloud/authentication/), [payload indexing](https://qdrant.tech/documentation/manage-data/indexing/), [cluster lifecycle](https://qdrant.tech/documentation/cloud/create-cluster/), [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Troubleshoot configuration without exposing keys

1. Deploy the revision containing Fireworks support to the branch watched by Streamlit. Updating Secrets alone does not add a missing provider implementation.
2. Copy root-level settings from [the runtime template](qdrant.secrets.example.toml). Supply all three private credentials directly in Streamlit. Replace the publisher-only `QDRANT_API_KEY` entry with `SENTO_QDRANT_READ_ONLY_API_KEY` containing the app read-only key, and remove `SENTO_ACTIVE_COLLECTION` from app Secrets. Paste ordinary TOML: no trailing backslashes and URLs starting with `https://`.
3. Confirm the collection really contains the matching provider, model, 4096 dimensions, release ID and ready control record. Do not rename an OpenAI collection or merely edit its metadata to make it look compatible. If publication is needed, prepare and publish a new candidate with `--embedding-provider fireworks`; existing collections are never overwritten.
4. If `RAG CONFIGURATION ERROR` persists, inspect the app logs for `SENTO configuration error:`. Controlled messages identify missing settings or unsupported combinations without logging secret values. Unknown initialization errors remain generic. The public response remains generic so visitors do not see deployment details.
5. Save the updated Secrets, restart/redeploy as needed, and submit a new question. Old error messages remain in existing chat histories. Test a supported question and an unsupported question; inspect citations and coverage limitations.

For an offline maintainer check, explicitly export settings to an ignored private file such as `.streamlit/secrets.qdrant.toml`, never a tracked documentation template. This file is only a maintenance input; it is not uploaded automatically or read as deployed app configuration. With the runtime read-only key name, run:

```powershell
.\.venv\Scripts\python.exe tools/check_rag.py --runtime-settings --secrets-file .streamlit/secrets.qdrant.toml --config-only
.\.venv\Scripts\python.exe tools/check_rag.py --runtime-settings --secrets-file .streamlit/secrets.qdrant.toml --connection-only
```

`--config-only` makes zero network calls and validates structure/credential presence only. `--connection-only` makes a Qdrant read request but does not verify publication or model access. The full package health command above verifies release metadata and payloads without embedding or generation calls. None of these checks proves that the OpenAI model is available to the configured account. Keep the configured generation target until account access is verified; do not substitute an unrelated key or endpoint silently.
