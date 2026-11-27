# RAG setup and publication

**Selected deployment:** Qdrant Cloud. Follow [QDRANT_SETUP.md](QDRANT_SETUP.md) for the cluster, Streamlit Secrets, publishing credentials and release steps. The hosted Chat path is Qdrant-only; local Chroma remains a maintenance option. The active release path uses only the project's senator Markdown profiles.

Use Python 3.12 and install requirements.txt. The hosted Chat runtime reads its settings directly from root-level Streamlit Secrets and requires hosted embeddings/generation plus an existing Qdrant Cloud collection. It never reads a local secrets file or process environment, publishes evidence, or falls back to a local service. Local development and maintenance CLI configuration remain separate.

## Cloud settings

Enter root-level TOML values in the existing Streamlit app's Settings → Secrets. Never commit credentials. The deployed app reads its own Streamlit Secrets; an explicitly exported private TOML file may be used only for the maintenance checks below. Collection and release values stay blank until a Markdown release has been published and checked:

```toml
SENTO_VECTOR_BACKEND = "qdrant_cloud"
SENTO_QDRANT_URL = "https://d9b16d6d-d989-4240-9c8c-cecffb5f36e1.ap-northeast-2-0.aws.cloud.qdrant.io"
SENTO_QDRANT_COLLECTION = "<published collection>"
SENTO_RELEASE_ID = "<published release>"
SENTO_QDRANT_READ_ONLY_API_KEY = "<sento-app read-only key>"
SENTO_EMBEDDING_PROVIDER = "fireworks"
SENTO_EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
SENTO_EMBEDDING_DIMENSIONS = "4096"
SENTO_EMBEDDING_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_API_KEY = "<private Fireworks credential>"
SENTO_GENERATOR_PROVIDER = "openai"
OPENAI_API_KEY = "<private gateway credential>"
OPENAI_BASE_URL = "https://aiki-labs.up.railway.app/api/v1"
SENTO_OPENAI_MODEL = "gpt-6-luna"
SENTO_OPENAI_REASONING_EFFORT = "low"
SENTO_RETRIEVAL_TOP_K = "5"
SENTO_MAX_COSINE_DISTANCE = "0.65"
SENTO_GLOBAL_REQUEST_LIMIT = "30"
SENTO_GLOBAL_REQUEST_WINDOW_SECONDS = "60"
```

Model access must be checked before rollout. An existing gateway credential is not assumed to work at OpenAI. Embedding and generation endpoints are configured separately. A model or dimension change requires a newly embedded collection. Missing cloud settings, incompatible collections or service failures never fall back to local storage or Ollama.

The UI allows eight submitted questions per session per rolling minute. The cloud RAG pipeline also allows thirty retrieval requests per app process per rolling minute, shared across users on that process. Exact repeat answers are cached only in the current session for up to ten minutes and 32 entries, keyed by the question, subject, period, release, model, style and prompt version. Cache hits skip query embedding, Qdrant search and generation. The app-process limit resets on restart and is not a monthly spending cap; configure Fireworks and gateway usage limits and alerts separately.

The runtime uses application-supplied Fireworks or OpenAI vectors in Qdrant Cloud with cosine search, schema/release checks and read-only query methods. `SENTO_QDRANT_READ_ONLY_API_KEY` is the only Qdrant credential name accepted from app Secrets. Confirm it has the Read Cluster role in the Qdrant dashboard; runtime naming cannot verify a key's server-side permissions. Keep publisher credentials in a separate maintainer environment. The app reads Streamlit Secrets directly through `st.secrets`; see [Streamlit Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

## Prepare a Markdown release

The r3 package below has already been published and checked. These commands document its preparation; use a new release ID, collection and output directory for future publication.

The Markdown profiles are trusted project-curated content for this school assignment. Release preparation does not use `data/releases/demo-r1/review.json` and does not require a separate claim approval register. It includes every body section and claim from every Markdown file, preserves each claim's existing evidence status and available source metadata, and records source-file checksums. It does not infer periods or alter claim statuses.

PDF and HTML ingestion is disabled. The importer reads `.md` and `.markdown` files from `data/senator_records/` only. The default release includes every section for every subject found in those files. Use repeated `--subject` options only when preparing an intentional subset.

```sh
python tools/ingest_senator_records.py
python tools/prepare_markdown_release.py --embedding-provider fireworks \
  --release-id demo-md-fireworks-qwen3-8b-r3 \
  --collection sento_demo_md_fireworks_qwen3_8b_4096_r3 \
  --output-dir data/releases/demo-md-fireworks-qwen3-8b-r3
```

The builder reparses the current Markdown files and refuses a stale or modified processed-chunk file. The immutable package contains `chunks.jsonl` and `manifest.json`, including per-senator counts, section locators, source checksums, embedding identity, and `curation_status: project_curated`. It contains no reviewer or approval date. The empty `demo-r1/review.json` is retained as historical data and is not consulted by this path.

## Publish a new candidate

In the maintainer environment, use matching Qdrant deployment settings and set `SENTO_ACTIVE_COLLECTION` to the serving collection, or `NONE` for first publication. Set `SENTO_QDRANT_COLLECTION` to a new candidate name. Do not change the running app yet. See [QDRANT_SETUP.md](QDRANT_SETUP.md) for the separate publisher settings and exact commands.

```sh
python tools/index_corpus.py --chunks data/releases/demo-md-fireworks-qwen3-8b-r3/chunks.jsonl --release-manifest data/releases/demo-md-fireworks-qwen3-8b-r3/manifest.json --dry-run
python tools/index_corpus.py --chunks data/releases/demo-md-fireworks-qwen3-8b-r3/chunks.jsonl --release-manifest data/releases/demo-md-fireworks-qwen3-8b-r3/manifest.json --apply
python tools/check_rag.py --chunks data/releases/demo-md-fireworks-qwen3-8b-r3/chunks.jsonl --release-manifest data/releases/demo-md-fireworks-qwen3-8b-r3/manifest.json
```

Dry run validates local hashes, chunk IDs, counts and provider identity without connecting or embedding. Apply creates a Qdrant collection; it cannot overwrite an existing name or the declared active collection. Writes use batches of 32. Failed uploads remain building, which runtime rejects. Retry with a new candidate name and matching manifest; never refresh a live collection with a subset. The checker paginates IDs and verifies complete metadata without paid model calls. Ready means upload verification passed; it does not mean independent fact-checking or answer-quality evaluation.

Before enabling the release in Streamlit, evaluate supported and unsupported questions across included subjects in English, Filipino and Taglish. Check sentence-level support, subject isolation, stale/current distinctions, citations, outages and prompt injection. Then set the new collection/release in app Secrets. Retain the prior collection for rollback and restore its matching collection, release and model settings together if needed.

## Local development

Local indexing remains available to maintainers through the CLI and Ollama. The Streamlit Chat adapter intentionally does not use local settings or indexes; it requires Streamlit Secrets and Qdrant/OpenAI cloud services.

```sh
ollama pull embeddinggemma
python tools/index_corpus.py
```

Bare indexing is a local maintenance command that replaces stale rows. Generated files are ignored by Git and are not deployed with the repository.

## Data flow and operation

Questions go to the configured embedding API (Fireworks for the selected release); vectors and subject/period filters go to Qdrant Cloud. Publication sends project-curated Markdown passages to the configured embedding provider, then stores passages, vectors and citation metadata in Qdrant. The selected configuration sends the current question, selected subject and bounded evidence to the owner-supplied gateway at https://aiki-labs.up.railway.app/api/v1 using Chat Completions, a 45-second timeout, no retries and a 700-token output cap. Direct OpenAI configurations use Responses with store=False. The gateway path does not send the Responses reasoning-effort option or a store flag. This does not guarantee zero provider retention. Earlier session messages are not sent; questions must stand on their own. No transcript, account, analytics or telemetry storage was added.

Cached services share clients/connections only. Histories and answer-cache entries remain session-local, separate per senator in chat history, and may reset on disconnect or restart. Explicit handoffs make no embedding/generation call and do not transfer history or resubmit questions. Subjects without eligible indexed claims have no retrieved passages; missing evidence is not proof of inactivity. Citation validation verifies marker/provenance integrity, not semantic entailment.

Configure spending limits and alerts privately and name a maintenance/correction owner. Test cloud wake, restart and another-device access with the laptop off before declaring deployment complete. See [DEPLOYMENT_STATUS.md](DEPLOYMENT_STATUS.md) for outstanding gates.
