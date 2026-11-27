from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import sys

import pytest
from qdrant_client import QdrantClient, models

from ai.qdrant_store import (QdrantVectorStore, RELEASE_POINT_ID, point_id,
                             publish_qdrant_release, verify_qdrant_collection)
from ai.rag_config import RAGConfig
from ai.rag_factory import build_default_pipeline, QdrantRetriever
from ai.releases import prepare_release, validate_package, publish_release, checksum
from ai.vector_store import VectorStoreError
from tests.test_cloud_deployment import reviewed

pytestmark = pytest.mark.filterwarnings('ignore:Payload indexes have no effect in the local Qdrant')


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    for key, value in {
        'SENTO_VECTOR_BACKEND': 'qdrant_cloud', 'SENTO_QDRANT_URL': 'https://test.cloud.qdrant.io',
        'SENTO_QDRANT_COLLECTION': 'sento_test_release', 'SENTO_RELEASE_ID': 'test-r1',
        'QDRANT_API_KEY': 'fake-test-key', 'OPENAI_API_KEY': 'fake-test-key',
        'SENTO_GENERATOR_PROVIDER': 'openai', 'SENTO_ACTIVE_COLLECTION': 'NONE',
        'SENTO_CHROMA_PATH': str(tmp_path / 'must-not-exist'),
    }.items():
        monkeypatch.setenv(key, value)
    for key in ['CHROMA_API_KEY', 'CHROMA_TENANT', 'CHROMA_DATABASE', 'SENTO_CHROMA_COLLECTION']:
        monkeypatch.delenv(key, raising=False)
    return RAGConfig.load()


@pytest.fixture
def embedder():
    return NS(provider_name='openai', model_name='text-embedding-3-small',
              embed=lambda texts: [[1.] + [0.] * 1535 for _ in texts])


@pytest.fixture
def published(reviewed, cloud, embedder):
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    rows = validate_package(selected, manifest, cloud)
    client = QdrantClient(':memory:')
    publish_release(rows, manifest, cloud, client=client, embedder=embedder)
    yield client, rows, manifest
    client.close()


def test_roundtrip_preserves_sources_subjects_and_periods(published, cloud, embedder):
    client, rows, _ = published
    store = QdrantVectorStore(embedder, cloud, client=client)
    assert store.count() == 2
    assert client.count(cloud.collection).count == 3  # two evidence points plus control metadata
    results = store.retrieve('Education', 'alpha', congress_id='20')
    assert [r['senator_id'] for r in results] == ['alpha']
    assert results[0]['citation_sources'] == rows[0]['citation_sources']
    assert results[0]['retrieval']['distance'] == pytest.approx(0.)
    assert store.retrieve('Education', 'alpha', congress_id='19') == []
    assert store.retrieve('Education', 'alpha', relevant_period='19th Congress') == []
    assert store.retrieve('Education', 'uncovered') == []
    verify_qdrant_collection(client, cloud.collection, rows)


def test_existing_collection_is_never_overwritten(published, cloud, embedder):
    client, rows, manifest = published
    with pytest.raises(VectorStoreError):
        publish_release(rows, manifest, cloud, client=client, embedder=embedder)
    verify_qdrant_collection(client, cloud.collection, rows)


def test_active_collection_guard_runs_before_network(published, cloud, embedder, monkeypatch):
    _, rows, manifest = published
    monkeypatch.setenv('SENTO_ACTIVE_COLLECTION', cloud.collection)
    with pytest.raises(ValueError, match='never overwrite'):
        publish_qdrant_release(rows, manifest, cloud, client=object(), embedder=embedder)


def test_failed_embedding_keeps_candidate_inactive(reviewed, cloud, embedder):
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    rows = validate_package(selected, manifest, cloud)
    client = QdrantClient(':memory:')
    embedder.embed = lambda texts: [[1., 2.] for _ in texts]
    with pytest.raises(VectorStoreError, match='dimensions'):
        publish_release(rows, manifest, cloud, client=client, embedder=embedder)
    control = client.retrieve(cloud.collection, [RELEASE_POINT_ID])[0]
    assert control.payload['publication_status'] == 'building'
    with pytest.raises(VectorStoreError, match='incomplete'):
        QdrantVectorStore(embedder, cloud, client=client)
    client.close()


@pytest.mark.parametrize('change', [
    {'embedding_model': 'other'}, {'embedding_dimensions': 3}, {'release_id': 'other'},
    {'schema_version': 'old'}, {'publication_status': 'building'}, {'expected_rows': 999},
])
def test_release_identity_and_state_are_required(published, cloud, embedder, change):
    client, _, _ = published
    client.set_payload(cloud.collection, change, points=[RELEASE_POINT_ID])
    with pytest.raises(VectorStoreError):
        QdrantVectorStore(embedder, cloud, client=client)


def test_actual_vector_configuration_is_checked(published, cloud, embedder):
    client, _, _ = published
    with pytest.raises(VectorStoreError, match='dimensions'):
        QdrantVectorStore(embedder, replace(cloud, dimensions=3), client=client)


def test_missing_collection_and_provider_outages_are_controlled(cloud, embedder):
    client = QdrantClient(':memory:')
    with pytest.raises(VectorStoreError, match='could not be opened'):
        QdrantVectorStore(embedder, cloud, client=client)
    client.close()


def test_factory_cloud_path_has_no_local_fallback_or_startup_embedding(published, cloud, embedder, monkeypatch):
    client, _, _ = published
    monkeypatch.setattr(RAGConfig, 'cloud_client', lambda self: client)
    monkeypatch.setattr(RAGConfig, 'embedder', lambda self: embedder)
    pipeline = build_default_pipeline(cloud)
    assert isinstance(pipeline.retriever, QdrantRetriever)
    assert pipeline.retriever.count_indexed() == 2
    assert not pipeline.retriever.persist_directory.exists()
    embedder.embed = lambda texts: pytest.fail('Uncovered subject must not trigger embedding')
    assert pipeline.retriever.retrieve('Question', 'uncovered') == []


def test_similarity_cutoff_and_untrusted_result_checks(published, cloud, embedder, monkeypatch):
    client, rows, _ = published
    store = QdrantVectorStore(embedder, cloud, client=client)
    payload = {**rows[0], 'record_kind': 'evidence'}
    wrong = {**payload, 'senator_id': 'beta'}
    other_period = {**payload, 'congress_id': '19'}
    other_release = {**payload, 'release_id': 'other'}
    def query_points(**kwargs):
        assert kwargs['score_threshold'] == pytest.approx(.35)
        constraints = {c.key: c.match.value for c in kwargs['query_filter'].must}
        assert constraints == {'record_kind': 'evidence', 'senator_id': 'alpha', 'curation_status': 'project_curated',
                               'release_id': cloud.release, 'congress_id': '20'}
        return NS(points=[NS(id=point_id(payload['index_id']), payload=p, score=s) for p, s in
                          [(payload, .8), (payload, .34), (payload, float('nan')), (wrong, .9),
                           (other_period, .9), (other_release, .9)]])
    monkeypatch.setattr(client, 'query_points', query_points)
    hits = store.retrieve('education', 'alpha', congress_id='20')
    assert len(hits) == 1
    assert hits[0]['retrieval']['distance'] == pytest.approx(.2)


@pytest.mark.parametrize('url', ['http://example.com', 'https://key@example.com', 'https://example.com/path',
                                 'https://example.com?key=secret', 'https://example.com:broken'])
def test_invalid_qdrant_endpoint_is_rejected(cloud, monkeypatch, url):
    monkeypatch.setenv('SENTO_QDRANT_URL', url)
    with pytest.raises(ValueError, match='HTTPS endpoint'):
        RAGConfig.load()


def test_missing_qdrant_credentials_never_fall_back(cloud, monkeypatch):
    monkeypatch.delenv('QDRANT_API_KEY')
    with pytest.raises(ValueError, match='incomplete'):
        RAGConfig.load()
    assert RAGConfig.load(require_credentials=False).backend == 'qdrant_cloud'


def test_qdrant_dry_run_has_zero_provider_access(reviewed, cloud, monkeypatch, tmp_path, capsys):
    from tools.index_corpus import main
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    chunks_path, review_path, manifest_path = [tmp_path / name for name in ['chunks.jsonl', 'review.json', 'manifest.json']]
    chunks_path.write_text(''.join(json.dumps(c) + '\n' for c in selected), encoding='utf-8')
    review_path.write_text(json.dumps(review), encoding='utf-8')
    manifest.update(chunks_sha256=checksum(chunks_path), review_sha256=checksum(review_path))
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(RAGConfig, 'cloud_client', lambda self: pytest.fail('Offline check connected'))
    monkeypatch.setattr(RAGConfig, 'embedder', lambda self: pytest.fail('Offline check built model client'))
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.delenv('QDRANT_API_KEY')
    monkeypatch.setattr(sys, 'argv', ['index_corpus', '--chunks', str(chunks_path), '--release-manifest', str(manifest_path), '--dry-run'])
    assert main() == 0
    assert 'offline_preflight_passed' in capsys.readouterr().out


def test_connection_check_needs_no_models_and_prints_no_key(cloud, monkeypatch, capsys):
    from tools.check_rag import main
    calls = []
    client = NS(get_collections=lambda: calls.append('read'), close=lambda: calls.append('close'))
    monkeypatch.setattr(RAGConfig, 'cloud_client', lambda self: client)
    monkeypatch.setattr(RAGConfig, 'embedder', lambda self: pytest.fail('Connection check must not build model client'))
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.setattr(sys, 'argv', ['check_rag', '--connection-only'])
    assert main() == 0
    assert calls == ['read', 'close']
    output = capsys.readouterr().out
    assert 'connection_ok' in output and 'fake-test-key' not in output


def test_explicit_toml_loading_never_echoes_invalid_secret(tmp_path, capsys):
    from tools.runtime_settings import load_settings
    settings = tmp_path / 'private.toml'
    settings.write_text('QDRANT_API_KEY = private-secret-invalid-toml', encoding='utf-8')
    with pytest.raises(ValueError) as exc:
        load_settings(settings)
    assert 'private-secret' not in str(exc.value)
    with pytest.raises(ValueError, match='.env is not loaded'):
        load_settings(tmp_path / '.env')
