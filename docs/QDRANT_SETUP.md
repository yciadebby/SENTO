# Qdrant deployment for SEN’TO

Qdrant replaces Chroma Cloud for the selected deployment. Local Chroma development and the Chroma Cloud adapter remain available. The user supplied a healthy `sento-demo` cluster and active `sento-publisher` (manage) and `sento-app` (read) keys in a dashboard screenshot. This is account-setup evidence, not an authenticated application check.

Endpoint: `https://d9b16d6d-d989-4240-9c8c-cecffb5f36e1.ap-northeast-2-0.aws.cloud.qdrant.io`

## Runtime settings in Streamlit Cloud

The hosted chatbot reads its RAG configuration directly from root-level values in the deployed app's Streamlit Settings → Secrets. It does not read `.env`, local TOML files, or process environment variables for chatbot settings. Use [the settings template](qdrant.secrets.example.toml) as a checklist, not as a runtime file. Paste the non-secret settings there and enter credentials directly in Streamlit Secrets.

Set `SENTO_QDRANT_READ_ONLY_API_KEY` to the full `sento-app` read-only key, `OPENAI_API_KEY` to the OpenAI project key for chat generation, and `FIREWORKS_API_KEY` for embeddings. Do not put the `sento-publisher` manage key or `QDRANT_API_KEY` in app Secrets. The app passes only the separately named read-only secret to the Qdrant client; confirm the key's Read Cluster role in the Qdrant dashboard. Keep the publisher key in the maintainer's separate publishing environment.

Leave `SENTO_QDRANT_COLLECTION` and `SENTO_RELEASE_ID` unset until a project-curated Markdown release has been published and checked. Missing cloud settings fail closed; the chat runtime cannot switch to local Chroma or Ollama.

The owner previously reported a successful connection-only read using the app key. That verified authentication at that time, not a published collection, write restrictions, or OpenAI model access.

## Prepare Markdown evidence for publication

The chatbot release uses only `data/senator_records/*.md` and `*.markdown`; PDF and HTML ingestion is disabled. No separate review register is needed. It includes every body section and claim, regardless of `Evidence_Status`, preserves the status and source metadata written in each file, and records the source-file checksum. Follow [RAG_SETUP.md](RAG_SETUP.md), then run:

```powershell
.\.venv\Scripts\python.exe tools/ingest_senator_records.py
.\.venv\Scripts\python.exe tools/prepare_markdown_release.py --release-id demo-md-r1 --collection sento_demo_md_r1_openai_te3small_1536 --output-dir data/releases/demo-md-r1
```

The package is created under `data/releases/demo-md-r1/`; use a new release ID, collection name and output directory for later revisions. The empty `data/releases/demo-r1/review.json` is historical and is not read by this workflow. No source file is edited or promoted by the exporter.

## Publish using the maintenance key

For the maintainer workflow, create a separate ignored publisher settings file. Enter the `sento-publisher` manage key as `QDRANT_API_KEY` and `FIREWORKS_API_KEY` for embeddings. Add `SENTO_ACTIVE_COLLECTION = "NONE"` for the first publication; for future releases use the actual serving collection name. Set the candidate collection/release to the manifest's matching values. Keep this publisher file out of Streamlit Secrets; the app uses only `SENTO_QDRANT_READ_ONLY_API_KEY`.

```powershell
.\.venv\Scripts\python.exe tools/index_corpus.py --secrets-file .streamlit/secrets.publisher.toml --chunks data/releases/demo-md-r1/chunks.jsonl --release-manifest data/releases/demo-md-r1/manifest.json --dry-run
.\.venv\Scripts\python.exe tools/index_corpus.py --secrets-file .streamlit/secrets.publisher.toml --chunks data/releases/demo-md-r1/chunks.jsonl --release-manifest data/releases/demo-md-r1/manifest.json --apply
.\.venv\Scripts\python.exe tools/check_rag.py --secrets-file .streamlit/secrets.qdrant.toml --chunks data/releases/demo-md-r1/chunks.jsonl --release-manifest data/releases/demo-md-r1/manifest.json
```

Dry run is offline and does not require model or database credentials. Apply sends Markdown text to OpenAI for paid embeddings and uploads vectors and evidence to Qdrant. It creates a new collection only; an existing name fails. Never use `recreate_collection` or replace the active release. Failed uploads stay inactive; retry with a new candidate collection and matching package. The full health check compares saved IDs and complete payloads against the package, with no model calls.

## Storage and retrieval contract

- One cosine collection per release, using vectors from the model and dimensions recorded in that release's manifest. Qdrant inference is not used; never mix models in one collection.
- Deterministic UUIDs map the existing `chunk_id::senator_id` row IDs to Qdrant points. Original IDs remain in each payload, along with complete text, citation sources, raw periods and normalized Congress.
- Keyword indexes are created before upload for `record_kind`, `senator_id`, `curation_status`, `congress_id`, `relevant_period`, and `release_id`. Original claim evidence statuses are preserved in payloads and are not used as an inclusion filter.
- One reserved control point stores release/model/schema identity, curation status, counts and `building`/`ready` state. It is excluded from every evidence query and from displayed coverage counts. The dashboard's total point count is therefore **expected evidence rows plus one**.
- Runtime opens existing collections and validates identity, dimensions, cosine metric, ready state and total count. It has no write methods. Every query filters evidence kind, project curation, release and selected senator, plus the requested period. Returned payloads are checked again; the pipeline independently checks subject and Congress.
- Qdrant cosine similarity is converted to SEN’TO's distance as `1 - similarity`; the existing maximum distance of 0.65 corresponds to a minimum similarity of 0.35. Evaluate the cutoff against reviewed questions before release.
- Provider connections are cached without histories. Exact repeat answers are cached only in the current session for ten minutes, up to 32 entries. The UI allows eight submissions per session/minute; cloud retrieval allows 30 requests per running app process/minute. These process/session controls do not replace OpenAI project spending limits and alerts. Read/write requests are bounded by a 45-second client timeout. Maintenance scans use pagination; uploads use batches of 32. Existing conservative application payload limits remain in force; these are not advertised as Qdrant service quotas.

## Deploy only after live review

Verify the configured Responses model and reasoning option against the OpenAI account before live evaluation. Complete the supported/unsupported question set and sentence-support review. Then deploy the code to the branch watched by the existing Streamlit app, using `app.py` and Python 3.12. Enter the runtime template's values in Streamlit Settings → Secrets using the **read-only** key and the published collection/release. Do not enter the publisher key. The app sends the current question and retrieved evidence to OpenAI for generation; query embeddings also use OpenAI, while vectors and source payloads live in Qdrant Cloud.

Test another device with the laptop off, cloud restart, and sleep/wake behavior. Keep the previous release and matching settings for rollback. Free Qdrant clusters can suspend after one week without activity and be deleted after four weeks without reactivation, so retain the versioned reviewed package for recovery. No account signup or current cloud retention promise replaces those acceptance checks.

References: [Qdrant authentication](https://qdrant.tech/documentation/cloud/authentication/), [payload indexing](https://qdrant.tech/documentation/manage-data/indexing/), [cluster lifecycle](https://qdrant.tech/documentation/cloud/create-cluster/), [Streamlit Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
