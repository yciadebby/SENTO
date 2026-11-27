"""Read-only Qdrant evidence access and create-only release publication.

One reserved control point holds release identity/state. Queries explicitly filter
for evidence points, so control data is never supplied to generation.
"""
from copy import deepcopy
import math
import os
from uuid import NAMESPACE_URL, uuid5

from ai.vector_store import VectorStoreError


RELEASE_POINT_ID = str(uuid5(NAMESPACE_URL, 'sento:release:metadata:v1'))
FILTER_FIELDS = ('record_kind', 'senator_id', 'verification_status', 'curation_status', 'congress_id', 'relevant_period', 'release_id')


def point_id(index_id):
    return str(uuid5(NAMESPACE_URL, 'sento:evidence:' + index_id))


def evidence_payload(record):
    return {**deepcopy(record), 'record_kind': 'evidence'}


def filters(**values):
    from qdrant_client import models
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))
                               for key, value in values.items() if value is not None])


def checked_vectors(vectors, count, dimensions):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise VectorStoreError('Embedding count does not match publication records.')
    for vector in vectors:
        if (not isinstance(vector, list) or len(vector) != dimensions
                or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in vector)):
            raise VectorStoreError('Embedding dimensions or values are invalid.')
    return vectors


class QdrantVectorStore:
    """Opens an existing release only; has no collection creation or write methods."""

    def __init__(self, embedder, config, *, client=None):
        from qdrant_client import models
        self.embedder = embedder
        self.config = config
        self.collection_name = config.collection
        self.client = client if client is not None else config.cloud_client()
        try:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            if not isinstance(vectors, models.VectorParams) or vectors.size != config.dimensions or vectors.distance != models.Distance.COSINE:
                raise VectorStoreError('Qdrant vector dimensions or distance metric do not match the release.')
            points = self.client.retrieve(self.collection_name, ids=[RELEASE_POINT_ID], with_payload=True, with_vectors=False)
            metadata = points[0].payload if len(points) == 1 else None
            if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in config.metadata().items()):
                raise VectorStoreError('Qdrant release, schema or embedding identity does not match configuration.')
            if metadata.get('record_kind') != 'release' or metadata.get('publication_status') != 'ready':
                raise VectorStoreError('Qdrant release publication is incomplete.')
            if (metadata.get('source_format') != 'senator_markdown'
                    or metadata.get('curation_status') not in {'project_curated', 'human_reviewed'}):
                raise VectorStoreError('Qdrant release is not a supported curated Markdown release.')
            counts = metadata.get('expected_counts')
            if (not isinstance(counts, dict) or not counts
                    or any(not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool) or v <= 0 for k, v in counts.items())
                    or sum(counts.values()) != metadata.get('expected_rows')):
                raise VectorStoreError('Qdrant release coverage metadata is invalid.')
            if getattr(embedder, 'provider_name', None) != config.embedding_provider or embedder.model_name != config.embedding_model:
                raise VectorStoreError('Qdrant query embedder does not match the stored model.')
            if self.client.count(self.collection_name, exact=True).count != metadata['expected_rows'] + 1:
                raise VectorStoreError('Qdrant point count does not match the published release.')
            self.metadata = deepcopy(metadata)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError('The configured Qdrant release could not be opened.') from None

    def count(self):
        try:
            return self.client.count(self.collection_name, exact=True,
                count_filter=filters(record_kind='evidence', release_id=self.config.release)).count
        except Exception:
            raise VectorStoreError('The Qdrant evidence service is unavailable.') from None

    def retrieve(self, query, senator_id, top_k=5, relevant_period=None, congress_id=None):
        if not query or not query.strip() or top_k <= 0 or senator_id not in self.metadata['expected_counts']:
            return []
        try:
            vector = checked_vectors(self.embedder.embed([query]), 1, self.config.dimensions)[0]
            response = self.client.query_points(
                collection_name=self.collection_name, query=vector, limit=min(top_k, 5),
                query_filter=filters(record_kind='evidence', senator_id=senator_id,
                    curation_status='project_curated', release_id=self.config.release,
                    relevant_period=relevant_period, congress_id=congress_id),
                # Chroma's cosine distance is 1 - Qdrant's cosine similarity.
                score_threshold=1.0 - self.config.max_distance, with_payload=True, with_vectors=False)
            records = []
            for hit in response.points:
                record = hit.payload
                if not isinstance(record, dict) or not isinstance(hit.score, (int, float)) or not math.isfinite(hit.score):
                    continue
                distance = 1.0 - hit.score
                if (record.get('record_kind') != 'evidence' or record.get('senator_id') != senator_id
                        or record.get('curation_status') != 'project_curated' or record.get('release_id') != self.config.release
                        or (congress_id and record.get('congress_id') != congress_id)
                        or (relevant_period and record.get('relevant_period') != relevant_period)
                        or not isinstance(record.get('text'), str) or not record['text'].strip()
                        or not isinstance(record.get('index_id'), str)
                        or str(hit.id) != point_id(record['index_id']) or distance > self.config.max_distance):
                    continue
                record = deepcopy(record)
                record['retrieval'] = {'method': 'qdrant_cosine', 'distance': distance,
                    'embedding_model': self.embedder.model_name, 'index_id': record['index_id']}
                records.append(record)
            return records
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError('The Qdrant evidence search is unavailable.') from None


def verify_qdrant_collection(client, collection_name, rows):
    """Scan bounded pages and compare exact IDs/payloads; never call a model."""
    expected = {point_id(r['index_id']): evidence_payload(r) for r in rows}
    observed, seen_offsets = {}, set()
    offset = None
    while True:
        page, next_offset = client.scroll(collection_name, limit=256, offset=offset,
                                          with_payload=True, with_vectors=False)
        for point in page:
            key = str(point.id)
            if key == RELEASE_POINT_ID:
                continue
            if key in observed:
                raise VectorStoreError('Duplicate point in publication scan.')
            observed[key] = point.payload
        if next_offset is None:
            break
        if str(next_offset) in seen_offsets:
            raise VectorStoreError('Repeated Qdrant pagination cursor.')
        seen_offsets.add(str(next_offset))
        offset = next_offset
    if observed != expected or client.count(collection_name, exact=True).count != len(rows) + 1:
        raise VectorStoreError('Qdrant publication IDs, payloads or counts do not match the release.')


def publish_qdrant_release(rows, manifest, config, *, client=None, embedder=None):
    """Create a fresh candidate; leave failed builds inactive and preserve old collections."""
    from qdrant_client import models
    from ai.releases import require, preflight_rows
    require(config.backend == 'qdrant_cloud', 'Qdrant publication requires qdrant_cloud configuration.')
    active = os.environ.get('SENTO_ACTIVE_COLLECTION')
    require(active and active != config.collection, 'Identify the active collection or NONE; never overwrite it.')
    preflight_rows(rows)
    client = client if client is not None else config.cloud_client()
    embedder = embedder if embedder is not None else config.embedder()
    metadata = {**config.metadata(), 'record_kind': 'release', 'publication_status': 'building',
                'expected_counts': manifest['expected_counts'], 'expected_rows': manifest['expected_rows'],
                'source_format': manifest['source_format'], 'curation_status': manifest['curation_status'],
                'prepared_at': manifest.get('prepared_at', ''),
                'reviewed_at': manifest.get('reviewed_at', '')}
    try:
        # Never use recreate_collection. The server rejects an existing name.
        created = client.create_collection(config.collection,
            vectors_config=models.VectorParams(size=config.dimensions, distance=models.Distance.COSINE))
        if not created:
            raise VectorStoreError('Qdrant did not confirm new collection creation.')
        for key in FILTER_FIELDS:
            client.create_payload_index(config.collection, field_name=key, field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
        control_vector = [0.] * config.dimensions
        client.upsert(config.collection, points=[models.PointStruct(id=RELEASE_POINT_ID, vector=control_vector, payload=metadata)], wait=True)
        for offset in range(0, len(rows), 32):
            batch = rows[offset:offset + 32]
            vectors = checked_vectors(embedder.embed([r['text'] for r in batch]), len(batch), config.dimensions)
            client.upsert(config.collection, points=[models.PointStruct(id=point_id(r['index_id']), vector=vector, payload=evidence_payload(r))
                for r, vector in zip(batch, vectors)], wait=True)
        verify_qdrant_collection(client, config.collection, rows)
        client.overwrite_payload(config.collection, payload={**metadata, 'publication_status': 'ready'}, points=[RELEASE_POINT_ID], wait=True)
        QdrantVectorStore(embedder, config, client=client)
    except VectorStoreError:
        raise
    except Exception as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        # Provider errors are sanitized: report class/status only, never headers,
        # request bodies or credentials that SDK exceptions may contain.
        detail = f"{type(exc).__name__}" + (f" (HTTP {status})" if status else '')
        raise VectorStoreError(f'Qdrant publication failed: {detail}. The active release was not switched.') from None
    return {'indexed': len(rows), 'collection': config.collection, 'status': 'ready_for_live_review'}
