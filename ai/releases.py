"""Project-curated Markdown release preparation and immutable cloud publication helpers."""
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re

from ai.validator import is_safe_source_url
from ai.vector_store import _prepare_chunks, _metadata_for, VectorStoreError


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def valid_date(value):
    try:
        date.fromisoformat(value)
        return True
    except (ValueError, TypeError):
        return False


def prepare_release(chunks, review, source_root):
    """Approval is a human assertion bound to exact input bytes, never inferred from VERIFIED."""
    require(review.get('release_id') and review.get('collection'), 'Release ID and collection are required.')
    subjects = review.get('subjects', [])
    require(len(subjects) == 2 and len(set(subjects)) == 2, 'The pilot must identify exactly two subjects.')
    require(valid_date(review.get('reviewed_at')), 'A dated review is required.')
    scope = review.get('scope_review', [])
    require(len(scope) == 2 and {s.get('senator_id') for s in scope} == set(subjects)
            and {s.get('bloc') for s in scope} == {'majority', 'minority'},
            'Pilot selection requires reviewed majority/minority scope evidence.')
    for item in scope:
        require(item.get('reviewer') and valid_date(item.get('reviewed_at')) and valid_date(item.get('as_of')),
                'Scope classification requires a reviewer and dated as-of check.')
        source = item.get('source', {})
        require(all(source.get(k) for k in ('title', 'issuing_body', 'source_url', 'retrieved_at', 'relevant_period', 'locator'))
                and is_safe_source_url(source.get('source_url')) and source.get('verification_status') == 'verified',
                'Scope classification requires traceable original-source provenance.')
    decisions = review.get('decisions', [])
    approved = [d for d in decisions if d.get('decision') == 'approved']
    require(approved, 'No human-approved claims are available. Research labels alone cannot be published.')
    by_key = {}
    for decision in approved:
        key = (decision.get('senator_id'), decision.get('claim_id'))
        require(key not in by_key, 'Duplicate approval.')
        require(key[0] in subjects and key[1], 'Approval is outside the pilot scope.')
        require(decision.get('reviewer') and valid_date(decision.get('reviewed_at')), 'Reviewer and review date are required.')
        require(re.fullmatch(r'[0-9a-f]{64}', decision.get('source_sha256', '')), 'Approval must bind the source checksum.')
        require(decision.get('congress_id') in {'19', '20', 'historical'}, 'Review must normalize Congress or label historical evidence.')
        require(decision.get('period_basis'), 'Record the source-supported basis for the normalized period.')
        by_key[key] = decision
    selected, matched = [], set()
    for original in chunks:
        key = (original.get('senator_id'), original.get('claim_id'))
        if key not in by_key:
            continue
        decision = by_key[key]
        require(original.get('source_sha256') == decision['source_sha256'], 'Reviewed claim changed; review again.')
        filename = original.get('source_file', '')
        require(filename and Path(filename).name == filename, 'Invalid source filename.')
        source = Path(source_root) / filename
        require(source.is_file() and checksum(source) == decision['source_sha256'], 'Source file does not match reviewed bytes.')
        require(original.get('verification_status') == 'verified', 'An approved claim lacks complete importer provenance.')
        associations = set(original.get('senator_ids', [])) | {original.get('senator_id')}
        require(associations <= set(subjects), 'A claim has unreviewed subject associations.')
        require(all((subject, original['claim_id']) in by_key for subject in associations), 'Every shared subject association requires approval.')
        sources = original.get('citation_sources', [])
        require(sources, 'Every claim needs citations.')
        checked = decision.get('sources', [])
        for citation in sources:
            require(all(citation.get(k) for k in ('title', 'issuing_body', 'source_url', 'retrieved_at', 'relevant_period', 'locator')), 'Incomplete source provenance.')
            require(citation.get('verification_status') == 'verified' and is_safe_source_url(citation['source_url']), 'Unsafe or unverified source.')
            require(any(s.get('source_url') == citation['source_url'] and s.get('locator') == citation['locator'] for s in checked), 'Human review must cover every original source and exact locator.')
        record = {**original, 'congress_id': decision['congress_id'], 'reviewed_at': decision['reviewed_at'],
                  'release_id': review['release_id'], 'curation_status': 'project_curated'}
        selected.append(record)
        matched.update((subject, original['claim_id']) for subject in associations)
    require(matched == set(by_key), 'Some approved claims are absent from the supplied chunks.')
    rows = _prepare_chunks(selected)
    counts = dict(Counter(r['senator_id'] for r in rows))
    require(set(counts) == set(subjects), 'Both pilot subjects need reviewed evidence.')
    require(len({c['chunk_id'] for c in selected}) == len(selected), 'Duplicate chunks are not allowed.')
    preflight_rows(rows)
    manifest = {'release_id': review['release_id'], 'collection': review['collection'], 'schema_version': '2',
                'source_format': 'senator_markdown', 'curation_status': 'human_reviewed',
                'embedding_provider': 'openai', 'embedding_model': 'text-embedding-3-small',
                'embedding_dimensions': 1536, 'distance_metric': 'cosine', 'reviewed_at': review['reviewed_at'],
                'subjects': subjects, 'scope_review': scope, 'expected_counts': counts, 'expected_rows': len(rows),
                'approved_ids': sorted(c['chunk_id'] for c in selected),
                'source_checksums': {c['source_file']: c['source_sha256'] for c in selected},
                'importer_versions': sorted({c['extraction_version'] for c in selected})}
    return selected, manifest


def prepare_markdown_release(
    chunks,
    release_id,
    collection,
    source_root,
    subjects=None,
    embedding_provider='openai',
    embedding_model=None,
    embedding_dimensions=1536,
):
    """Build a package from every section in the project's curated Markdown corpus.

    The source files are treated as the project's chosen corpus. This records
    provenance and exact input bytes; it does not claim an independent review.
    """
    require(release_id and collection, 'Release ID and collection are required.')
    source_root = Path(source_root)
    from ai.senator_markdown import ingest_project_curated_markdown
    current_chunks = ingest_project_curated_markdown(source_root)
    def fingerprint(records):
        return {item.get('chunk_id'): json.dumps(item, ensure_ascii=False, sort_keys=True) for item in records}
    require(fingerprint(chunks) == fingerprint(current_chunks),
            'Processed chunks are stale or differ from the current Markdown files; rerun the importer.')
    chunks = current_chunks
    requested = set(subjects or [])
    curated_chunks = [c for c in chunks if c.get('curation_status') == 'project_curated']
    require(curated_chunks, 'No project-curated Markdown chunks are available for this release.')
    available = {c.get('senator_id') for c in curated_chunks}
    if requested:
        require(None not in requested and requested <= available, 'Requested subjects must exist in the Markdown corpus.')
        curated_chunks = [c for c in curated_chunks if c.get('senator_id') in requested]
    selected = []
    source_checksums = {}
    for original in curated_chunks:
        require(original.get('source_mime_type') == 'text/markdown', 'Only senator Markdown records may be released.')
        filename = original.get('source_file', '')
        require(filename and Path(filename).name == filename and Path(filename).suffix.casefold() in {'.md', '.markdown'},
                'Release inputs must be Markdown files inside the corpus directory.')
        source = source_root / filename
        require(source.is_file() and checksum(source) == original.get('source_sha256'),
                'A Markdown source changed after ingestion; rebuild the chunks.')
        citations = original.get('citation_sources', [])
        require(citations, 'A Markdown chunk must retain citation or profile-locator metadata.')
        # Preserve source-backed periods as entered. Do not infer a Congress.
        record = {**original, 'release_id': release_id}
        selected.append(record)
        source_checksums[filename] = original['source_sha256']
    ids = [c.get('chunk_id') for c in selected]
    require(all(ids) and len(ids) == len(set(ids)), 'Duplicate or missing Markdown chunk IDs.')
    rows = _prepare_chunks(selected)
    counts = dict(Counter(r['senator_id'] for r in rows))
    require(counts and (not requested or set(counts) == requested), 'Selected subjects need Markdown content.')
    preflight_rows(rows)
    embedding_model = embedding_model or ('sento-hash-ngram-v1' if embedding_provider == 'local_hash' else 'accounts/fireworks/models/qwen3-embedding-8b' if embedding_provider == 'fireworks' else 'text-embedding-3-small')
    manifest = {
        'release_id': release_id,
        'collection': collection,
        'schema_version': '2',
        'source_format': 'senator_markdown',
        'curation_status': 'project_curated',
        'prepared_at': date.today().isoformat(),
        'embedding_provider': embedding_provider,
        'embedding_model': embedding_model,
        'embedding_dimensions': embedding_dimensions,
        'distance_metric': 'cosine',
        'subjects': sorted(counts),
        'expected_counts': counts,
        'expected_rows': len(rows),
        'chunk_ids': sorted(ids),
        'source_checksums': source_checksums,
        'importer_versions': sorted({c['extraction_version'] for c in selected}),
    }
    return selected, manifest


def preflight_rows(rows):
    require(bool(rows), 'A release cannot be empty.')
    for row in rows:
        require(len(row['index_id'].encode('utf-8')) <= 128, 'Cloud ID exceeds 128 bytes.')
        require(len(row['text'].encode('utf-8')) <= 16384, 'Cloud document exceeds 16384 bytes.')
        require(all(len(str(v).encode('utf-8')) <= 8182 for v in _metadata_for(row).values()), 'Cloud metadata exceeds 8182 bytes; do not truncate evidence.')


def validate_package(chunks, manifest, config):
    require(manifest.get('collection') == config.collection and manifest.get('release_id') == config.release, 'Destination differs from the release manifest.')
    for key, value in config.metadata().items():
        require(manifest.get(key) == value, 'Release embedding/schema identity mismatch.')
    identity_list = manifest.get('chunk_ids') if manifest.get('curation_status') == 'project_curated' else manifest.get('approved_ids')
    require(sorted(c.get('chunk_id', '') for c in chunks) == identity_list, 'Release allowlist does not match chunks.')
    require(manifest.get('source_format') == 'senator_markdown'
            and manifest.get('curation_status') in {'project_curated', 'human_reviewed'},
            'Unsupported release source or curation status.')
    require(all(c.get('release_id') == config.release
                and (not c.get('congress_id') or c.get('congress_id') in {'19', '20', 'historical'})
                for c in chunks), 'Release identity or normalized period metadata is invalid.')
    require(all(c.get('curation_status') == 'project_curated'
                and c.get('source_mime_type') == 'text/markdown'
                and Path(c.get('source_file', '')).suffix.casefold() in {'.md', '.markdown'}
                and c.get('source_sha256') == manifest.get('source_checksums', {}).get(c.get('source_file'))
                for c in chunks), 'Release contains non-Markdown or changed source data.')
    for chunk in chunks:
        require(chunk.get('citation_sources'), 'Release chunk has no citation or source-locator metadata.')
    rows = _prepare_chunks(chunks)
    preflight_rows(rows)
    counts = dict(Counter(r['senator_id'] for r in rows))
    require(counts == manifest.get('expected_counts') and len(rows) == manifest.get('expected_rows'), 'Release counts differ from the manifest.')
    require(set(counts) == set(manifest.get('subjects', [])) and bool(counts), 'Invalid release subject scope.')
    return rows


def publish_release(rows, manifest, config, *, client=None, embedder=None):
    """Create only. Existing names fail; a partial upload is never promoted or overwritten."""
    import os
    if config.backend == 'qdrant_cloud':
        from ai.qdrant_store import publish_qdrant_release
        return publish_qdrant_release(rows, manifest, config, client=client, embedder=embedder)
    require(config.backend == 'chroma_cloud', 'Cloud publication requires cloud configuration.')
    active = os.environ.get('SENTO_ACTIVE_COLLECTION')
    require(active and active != config.collection, 'Set SENTO_ACTIVE_COLLECTION to the serving collection, or NONE for first publication; never overwrite it.')
    from chromadb import Schema, VectorIndexConfig
    metadata = {**config.metadata(), 'coverage_json': json.dumps(manifest['expected_counts'], separators=(',', ':')),
                'source_format': manifest['source_format'], 'curation_status': manifest['curation_status'],
                'prepared_at': manifest.get('prepared_at', ''),
                'reviewed_at': manifest.get('reviewed_at', ''), 'publication_status': 'building'}
    require(all(len(str(v).encode()) <= 256 for v in metadata.values()), 'Collection metadata exceeds cloud limits.')
    client = client or config.cloud_client()
    embedder = embedder or config.embedder()
    collection = client.create_collection(name=config.collection, embedding_function=None, metadata=metadata,
        schema=Schema().create_index(config=VectorIndexConfig(space='cosine')))
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        collection.add(ids=[r['index_id'] for r in batch], documents=[r['text'] for r in batch],
                       metadatas=[_metadata_for(r) for r in batch], embeddings=embedder.embed([r['text'] for r in batch]))
    verify_collection(collection, rows)
    collection.modify(metadata={**metadata, 'publication_status': 'ready'})
    return {'indexed': len(rows), 'collection': config.collection, 'status': 'ready_for_runtime_check'}


def verify_collection(collection, rows):
    expected = {r['index_id']: _metadata_for(r) for r in rows}
    observed = {}
    offset = 0
    while True:
        page = collection.get(limit=300, offset=offset, include=['metadatas'])
        ids = page.get('ids', [])
        observed.update(zip(ids, page.get('metadatas', [])))
        if len(ids) < 300:
            break
        offset += len(ids)
    require(observed == expected and collection.count() == len(rows), 'Uploaded IDs, provenance or counts do not match the release.')
