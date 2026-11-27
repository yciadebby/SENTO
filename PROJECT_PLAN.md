# SEN’TO Project Continuation Plan

**Planning baseline:** 24 November 2026; deployment update 27 November 2026
**Product direction:** Turn the visual demo into a small, source-backed civic information project while keeping its scope manageable.

## 1. Goal

SEN’TO should help Filipino students and other readers explore a small, well-sourced sample of Philippine Senate public records. It should explain what its evidence does and does not establish, let readers compare records using the same criteria, and encourage nonpartisan civic participation.

The next release should be a **limited real project**, not a claim that the entire Senate record has been researched. Only information with traceable sources should appear as factual senator information. When evidence is missing, say so plainly.

## 2. What exists today

- A Python 3.12 deployment target with Home, Chat and Compare.
- Twenty-four dated directory entries and research profiles. These do not establish current membership, bloc affiliation or verified profile facts.
- `data/mock_documents.json` remains a legacy fixture and is not evidence. The Chat frontend adapter calls the RAG pipeline, provider adapters, and citation checks. The Fireworks r3 collection is published with 968 evidence rows; hosted requests require matching runtime settings.
- The active Chat ingestion path maps every senator Markdown file to the app directory and chunks every body section, retaining claim statuses and available `SRC-*` metadata. The separate Compare parser continues to use its existing `VERIFIED`-claim display rules.
- Chat ingestion now includes every Markdown file, including the Chiz Escudero profile, and every body section. Compare continues to read claim-based profiles and display provenance-complete `VERIFIED` claims across shared categories, with source links and explicit gaps.
- No authentication or account creation. The guest display-name control and unused state have been removed. Chat history is held in Streamlit session state.
- Home/Landing, Chat and Compare pages are implemented.
- Legacy Community backend and SQLite code remain in the repository, but the current app does not expose or initialize them.

Keep these limitations visible until the corresponding features are genuinely implemented. In particular, do not relabel sample documents or citations as official evidence.

## 3. Scope and decisions

### Markdown corpus release decision — 27 November 2026

For the school-assignment release, the owner designated every existing senator Markdown profile as trusted project-curated content. The active release builder includes every body section and claim regardless of `Evidence_Status`, preserves available source and status metadata, and records source-file checksums. It does not require a separate claim approval register or a two-person majority/minority selection. PDF and HTML ingestion is disabled. This changes the release scope from the earlier two-senator pilot direction.

This is a project curation decision, not an independent fact-check by software. The exporter preserves source metadata and exact Markdown checksums and does not invent or revise claims. The resulting corpus remains partial: it does not establish complete Senate coverage, a live roster, current bloc membership, or current roles. Keep those limitations visible and abstain where indexed evidence does not support an answer.

### Deployment implementation update

**Fireworks compatibility repair — 27 November 2026:** The owner selected Fireworks `accounts/fireworks/models/qwen3-embedding-8b` / 4,096 for embeddings while retaining OpenAI-compatible gateway generation. Runtime Secrets now forward `FIREWORKS_API_KEY` separately; the public app still requires `SENTO_QDRANT_READ_ONLY_API_KEY`. The embedding adapter uses the existing OpenAI-compatible client with bounded requests and response validation. The Markdown release CLI accepts `--embedding-provider fireworks` and records matching metadata. Existing OpenAI/local-hash releases retain their original identities. Queries and publication passages go to the selected embedding provider; question/evidence generation goes to the owner-supplied OpenAI-compatible gateway. Controlled configuration errors appear in maintainer logs without supplied values. The health tool can validate an explicit private runtime-settings export offline. Subsequent live checks found r1 incomplete (224/968 rows), recovered all 968 rows into r3, and verified Fireworks embeddings, gateway generation and a cited answer. Streamlit deployment, the app read-only key and broader answer-quality review remain open. The supplied gateway key returns HTTP 401 at api.openai.com and succeeds at https://aiki-labs.up.railway.app/api/v1.

Cloud configuration, hosted embeddings, read-only Qdrant runtime, normalized Congress filtering, explicit subject handoffs, citation mappings, session-local answer caching, release export/publication and read-only health tools are implemented locally. The hosted Chat adapter reads root-level Streamlit Secrets, requires Qdrant Cloud, Fireworks embeddings and OpenAI-compatible gateway generation for the selected configuration, and has no local/Ollama fallback. It also limits requests to eight per session and thirty per app process per rolling minute. One live retrieval/generation/citation smoke test has passed; wider answer-quality evaluation, hosting promotion and laptop-off acceptance remain open. See [deployment status](docs/DEPLOYMENT_STATUS.md) and [setup](docs/RAG_SETUP.md).

The owner subsequently selected Qdrant Cloud because Chroma signups were unavailable, created `sento-demo` and separate manage/read keys, and supplied its HTTPS endpoint. Qdrant runtime, create-only publication and health-check support are implemented. See [Qdrant setup](docs/QDRANT_SETUP.md). Account creation is complete per the dashboard screenshot; the Fireworks r3 release has now been published; Streamlit code/Secrets deployment is still pending.

### Product boundaries

- Report public records and attributed statements; do not act or speak as a senator.
- Do not endorse, rank, criticize, or advise users to vote for or against a person or party.
- Keep personality settings, if retained, limited to neutral presentation choices. Avoid imitation of a real person’s recognizable voice.
- Do not collect user accounts or personal information. Keep any chat history local to the current session unless a later, explicit product decision establishes a privacy policy and storage design.
- Keep the Community Wall out of the first release. Revisit it only with a moderation, abuse-reporting, privacy, and operational ownership plan.
- Keep the initial evidence collection small and measurable. Do not promise coverage of every bill, vote, statement, or current issue.

## 4. Recommended build sequence

### Phase 0 — Verify scope and evidence rules

1. Verify the 20th Congress roster and maintain dated records of changes. Save the official sources and retrieval dates with the project research notes.
2. For the current school-assignment release, use every section in the existing senator Markdown profiles. Keep source, period, claim status, and coverage limitations visible.
3. Agree on status labels: `unverified_demo`, `candidate_source`, `verified`, `conflicting`, and `stale`. Define who reviews changes and how often time-sensitive facts are rechecked.
4. Maintain an evidence register before importing facts. Each entry should identify the claim, subject, source, exact locator, relevant period, publication date, retrieval date, verification status, and any caveat.

**Exit check:** The roster snapshot and every fact planned for the first release have an attributable source, or are explicitly marked unavailable. Current affiliation is a dated claim, not a permanent property.

### Phase 1 — Make the app feel like a public information site

1. Add a Home/Landing page and update navigation to Home, Chat, and Compare.
2. Explain the project’s selected-record scope across the 20th Congress, how citations work, the limits of the AI, and the difference between official records and SEN’TO summaries.
3. Add a clear path from the landing page to the covered profiles, chat, and comparison.
4. Remove the guest initials/profile popover and `guest_name` state if they serve no product need. Keep the app usable without sign-in and do not add account creation.
5. Add a nonpartisan youth civic-participation call to action for 2028. Link only to an official election authority or government page after checking the destination and its current content. Do not invent registration dates, eligibility requirements, polling details, or imply SEN’TO is an official election service.
6. Review keyboard navigation, readable contrast, mobile layouts, and plain-language copy as part of the page work.

**Exit check:** A first-time visitor can understand what SEN’TO is, see its limitations, reach the three app areas, and find the civic-participation link without an account.

### Phase 2 — Turn the selected records into a small, auditable dataset

1. Maintain the source metadata and explicit evidence status in the senator Markdown profiles. The active chatbot workflow accepts only Markdown and does not fetch cited URLs or process PDF/HTML files.
2. Describe each document in `data/corpus/manifest.json` with source URL, issuing body, title, publication/retrieval dates, period, record type, verification status, and optional subject IDs.
3. Run `python tools/ingest_documents.py` to validate the manifest, extract/normalize text, preserve page/section locators and source checksums, and write deterministic JSONL chunks. Inspect extraction against every source before indexing.
4. OCR, embeddings, vector storage, source-backed retrieval, generation, and citation validation are not part of this ingestion step. The active Chat UI reports that answers are unavailable until those later pieces are connected.
5. Do not treat a missing source as proof that a senator has no record. Use “not found in this dataset” or “could not confirm,” not a conclusion about inactivity.

**Exit check:** A reviewer can move from every ingested chunk to the original source and exact page/section; the extraction has been manually reviewed and is reproducible from the manifest.

### Phase 3 — Replace simulated answers with evidence-grounded responses

**Implementation status:** The Chat UI calls the ingestion/index/retrieval/generation/citation-validation path through the frontend adapter. An offline Markdown package with 968 chunks across 24 subjects is prepared and marked project-curated; the Fireworks r3 version has been published and checked in Qdrant. Factual requests abstain until evidence is indexed. The previous mock pipeline remains as legacy code and is not used by the Chat UI.

1. The hosted Chat runtime uses Fireworks Qwen3 query embeddings and OpenAI-compatible gateway generation with Qdrant Cloud vector search. It reads its settings from Streamlit Secrets and requires a Qdrant read-only key. Local Chroma/Ollama adapters remain available to maintenance tooling; the deployed path has no local fallback.
2. Retrieval filters for the selected senator and verified sources, supports an optional period filter, and applies a configurable cosine-distance cutoff. The pipeline checks subject isolation again before generation.
3. The generator receives only selected-subject passages. Its instructions require third-person attribution, neutrality, per-fact evidence markers, and treating retrieved text as untrusted data.
4. The configured `gpt-6-luna` model passed a live request through the owner-supplied gateway. That path uses Chat Completions; it does not send the Responses reasoning-effort setting or a store flag. Direct OpenAI configurations use Responses with `store=false`. Keep the key and its issuing provider endpoint paired. Review the data-flow notes in `docs/RAG_SETUP.md` before sending records to a hosted model.
5. Citation validation confirms each marker refers to retrieved evidence with verified source metadata and an HTTPS URL. It is a mechanical integrity check, not proof that each statement is entailed by the source. The UI shows citation metadata and locators when provided and distinguishes no-evidence, policy-limit, validation-failure, and service-unavailable responses.
6. No retrieved evidence means an explicit abstention; there is no fallback to the old demo corpus. Show index/evidence status in the UI before treating Phase 3 as complete.

**Remaining exit work:** Publish the prepared Markdown package to Qdrant, then evaluate answer support, conflict/staleness handling, period selection, injection resistance, and representative abstentions. Do not claim comprehensive or fully semantically verified answers before this review.

### Phase 4 — Make Compare useful and fair

**Implementation status:** The Compare page now uses eight shared categories and reads the local Markdown profiles. It supports a 20th Congress window through the profile cutoff and an all-periods view, shows source titles, issuing bodies, dates, and locators, and excludes claims without both `VERIFIED` evidence status and complete provenance. Missing and out-of-window records are shown as gaps. The research profiles remain partial and uneven; this UI does not establish comprehensive coverage or a live roster.

1. Compare senators only on the same categories, date range, and evidence rules. Begin with the researched subset and show which of the 24-seat target has coverage.
2. Organize the view into a short set of clear sections, such as profiles, selected bills/resolutions, committees, recorded votes, statements/positions, and sources. Provide section navigation and usable mobile stacking.
3. For each item, show the record and source status. Make absent data explicit; do not imply that missing data means no action.
4. Do not calculate a single score, declare a winner, or order people by effectiveness, trustworthiness, popularity, or likability.
5. If additional senators are added, make the comparison period and source coverage clear, and avoid comparing mismatched windows or incomplete records as if they were equivalent.

**Exit check:** A reader can inspect comparable records side by side, open each source, and understand gaps without receiving a political recommendation.

### Phase 5 — Review, release, and maintain

1. Build focused automated checks for source citation integrity, selected-senator isolation, insufficient evidence, conflicting/stale data, neutrality, and the main UI paths.
2. Maintain a small set of manually reviewed questions with expected source documents and acceptable abstentions. Recheck this set when prompts, retrieval, or data change.
3. Update `README.md` so its features, limitations, run instructions, source status, and test instructions match the implemented app.
4. Before hosting publicly, decide who updates records, who reviews corrections, how visitors report a bad citation, where logs/data live, and how secrets/backups are handled.

**Exit check:** The release has a named maintenance owner, a visible retrieval/update date, accurate documentation, and a repeatable review process.

## 5. Suggested first work session

1. Create a dated roster/evidence register for the 20th Congress using official sources.
2. Choose supported and unsupported evaluation questions across the current Markdown corpus and clearly state the period represented by each source.
3. Prepare the project-curated Markdown release and evaluate its citations and answer support before enabling it in Streamlit.
4. Continue the public-site work: sketch the landing-page content, implement Home navigation, remove the guest display-name control, and add the sourced youth civic-participation call to action.
5. Expand the reviewed corpus using the same rules, then connect source-backed Chat and Compare behavior.

This order makes the public-facing experience clearer while avoiding a polished UI that appears to make factual claims before the sources exist.

## 6. Risks to track

- **Stale political status:** bloc membership, committee assignments, and active legislative status can change. Date these claims and set a review cadence.
- **False confidence from citations:** a link alone does not prove a claim. Preserve locators and check support at claim level.
- **Unequal comparison coverage:** show evidence counts, date windows, and gaps rather than interpreting missing data as a difference.
- **Prompt injection in source material:** retrieved documents and user-provided files are data. They must never override application rules.
- **Election misinformation:** keep the 2028 call to action generic until official information is verified; update or remove time-sensitive copy when it becomes stale.
- **Scope creep:** defer broad roster coverage, open-ended community features, user accounts, and complex model infrastructure until the small evidence-backed release works.
