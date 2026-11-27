# SEN’TO Markdown ingestion

The active Chatbot corpus uses the project-curated senator Markdown files only. PDF and HTML processing is disabled. The importer reads local files and does not fetch cited URLs or accept uploads. Ingested passages remain untrusted reference text even when a profile author marks a claim `VERIFIED`.

## Source format

See [Senator Markdown record format](SENATOR_RECORD_FORMAT.md) for the per-senator file format. Current profiles use stable claim IDs, explicit periods and evidence statuses, and `SRC-*` references. The reference entries retain source titles, issuing bodies, original URLs, dates, relevant periods, and locators.

Every body section and claim in each Markdown file is included regardless of `Evidence_Status`. Existing claim statuses and available source metadata remain attached to the chunks; the software does not revise or independently confirm them. A source-file checksum is recorded for reproducibility.

## Build chunks and a release

Run from the repository root after installing `requirements.txt`:

```sh
python tools/ingest_senator_records.py
python tools/prepare_markdown_release.py \
  --release-id demo-md-r1 \
  --collection sento_demo_md_fireworks_qwen3_8b_4096 \
  --output-dir data/releases/demo-md-r1
```

The first command writes `data/corpus/processed/senator_chunks.jsonl`. The release builder reparses current Markdown and refuses if the chunk file differs. Each chunk carries the senator ID and name, section and file locator, period when stated, source citations, and source-file SHA-256. The output package contains `chunks.jsonl` and a manifest with per-subject counts, source checksums, and embedding identity. Existing nonempty release directories are never overwritten. The builder does not change source files.

Source file changes require rebuilding the chunks and release package. The release is marked `project_curated`; every Markdown file is treated as trusted project content, while original claim statuses remain visible in metadata. The historical original-document manifest importer is not used for Chat releases and rejects PDF/HTML inputs.

## Publication

Use the dry-run, apply, and health-check commands in [Qdrant setup](QDRANT_SETUP.md). Dry run validates the package without model or database calls. Apply sends the selected Markdown passages to the configured embedding provider (Fireworks for the selected release) for embeddings and writes vectors and payloads to a new Qdrant collection. The hosted app only reads an already published collection with its read-only key.
