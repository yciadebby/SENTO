import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from ai.embeddings import EmbeddingError, OpenAIEmbeddingProvider
from ai.generation import GenerationError, OpenAIResponsesProvider
from ai.rag_config import RAGConfig
from ai.rag_factory import build_default_pipeline
from ai.releases import prepare_release, validate_package, publish_release, preflight_rows, verify_collection
from ai.vector_store import ChromaVectorStore, VectorStoreError, _prepare_chunks
from ai.validator import format_citations
from backend.chatbot_service import ChatbotService
from backend.chat_scope import scope_response
from tests.test_vector_store import chunk, FakeEmbedder


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    settings = {'SENTO_VECTOR_BACKEND': 'chroma_cloud', 'SENTO_CHROMA_COLLECTION': 'sento_test_release',
                'SENTO_RELEASE_ID': 'test-r1', 'CHROMA_TENANT': 'test-tenant', 'CHROMA_DATABASE': 'test-db',
                'CHROMA_API_KEY': 'fake-key', 'OPENAI_API_KEY': 'fake-key', 'SENTO_GENERATOR_PROVIDER': 'openai',
                'SENTO_ACTIVE_COLLECTION': 'previous-release', 'SENTO_CHROMA_PATH': str(tmp_path / 'absent')}
    for k, v in settings.items():
        monkeypatch.setenv(k, v)
    return RAGConfig.load()


def test_hosted_embedding_orders_vectors_and_bounds_requests():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(data=[NS(index=1, embedding=[2.] * 1536), NS(index=0, embedding=[1.] * 1536)])
    provider = OpenAIEmbeddingProvider(api_key=None, base_url='https://api.openai.com/v1', client=NS(embeddings=NS(create=create)))
    assert [r[0] for r in provider.embed(['a', 'b'])] == [1., 2.]
    assert calls[0] == {'model': 'text-embedding-3-small', 'input': ['a', 'b'], 'dimensions': 1536}


@pytest.mark.parametrize('rows', [[], [NS(index=1, embedding=[0.] * 1536)],
    [NS(index=0, embedding=[0.] * 1535)], [NS(index=0, embedding=[float('nan')] * 1536)],
    [NS(index=0, embedding=['1'] * 1536)], [NS(index=0, embedding=[True] * 1536)]])
def test_embedding_rejects_malformed_responses(rows):
    provider = OpenAIEmbeddingProvider(api_key=None, base_url='https://api.openai.com/v1', client=NS(embeddings=NS(create=lambda **kw: NS(data=rows))))
    with pytest.raises(EmbeddingError):
        provider.embed(['a'])


def test_embedding_timeout_does_not_expose_credentials():
    def broken(**kwargs):
        raise TimeoutError('private-key')
    provider = OpenAIEmbeddingProvider(api_key=None, base_url='https://api.openai.com/v1', client=NS(embeddings=NS(create=broken)))
    with pytest.raises(EmbeddingError) as error:
        provider.embed(['a'])
    assert 'private-key' not in str(error.value)


def test_incomplete_generation_is_withheld():
    provider = OpenAIResponsesProvider(client=NS(responses=NS(create=lambda **kw: NS(status='incomplete', output_text='Partial claim [E1]'))))
    with pytest.raises(GenerationError, match='incomplete'):
        provider.complete('instructions', 'input')


def fake_collection(config, **overrides):
    metadata = {**config.metadata(), 'publication_status': 'ready', 'coverage_json': '{"alpha":1,"beta":1}', 'reviewed_at': '2026-11-26', **overrides}
    from chromadb import Schema, VectorIndexConfig
    return NS(metadata=metadata, schema=Schema().create_index(config=VectorIndexConfig(space='cosine')), count=lambda: 0)


def test_cloud_runtime_opens_only_existing_collection_without_local_gate(cloud, monkeypatch):
    calls = []
    def get_collection(**kwargs):
        calls.append(kwargs)
        return fake_collection(cloud)
    monkeypatch.setattr(RAGConfig, 'cloud_client', lambda self: NS(get_collection=get_collection))
    pipeline = build_default_pipeline(cloud)
    assert pipeline.retriever.retrieve('education', 'alpha') == []
    assert calls == [{'name': cloud.collection, 'embedding_function': None}]
    assert not pipeline.retriever.persist_directory.exists()
    with pytest.raises(VectorStoreError, match='read-only'):
        pipeline.retriever._get_store().index_documents([])


@pytest.mark.parametrize('overrides', [{'embedding_model': 'wrong'}, {'embedding_dimensions': 3}, {'release_id': 'other'}, {'schema_version': '1'}, {'publication_status': 'building'}])
def test_injected_collection_cannot_bypass_identity(cloud, overrides):
    with pytest.raises(VectorStoreError):
        ChromaVectorStore(cloud.embedder(), collection=fake_collection(cloud, **overrides), expected_metadata=cloud.metadata())


def test_wrong_saved_metric_is_rejected(cloud):
    collection = fake_collection(cloud)
    collection.schema.keys['#embedding'].float_list.vector_index.config.space = 'l2'
    with pytest.raises(VectorStoreError, match='cosine'):
        ChromaVectorStore(cloud.embedder(), collection=collection, expected_metadata=cloud.metadata())


def test_bad_cloud_settings_never_fall_back(cloud, monkeypatch):
    monkeypatch.delenv('CHROMA_TENANT')
    bot = ChatbotService()
    assert bot.ask('education', 'risa-hontiveros')['validation']['status'] == 'CONFIGURATION_ERROR'
    assert bot.pipeline is None


@pytest.mark.parametrize(('question', 'kind', 'targets'), [
    ('Tell me about Raffy Tulfo', 'SWITCH_SUBJECT', {'raffy-tulfo'}),
    ('Risa Hontiveros and Bam Aquino records', 'SWITCH_SUBJECT', {'paolo-benigno-aquino-iv'}),
    ('Tulfo bills', 'CLARIFY_SUBJECT', {'raffy-tulfo', 'erwin-tulfo'}),
    ('Cayetano records', 'CLARIFY_SUBJECT', {'alan-peter-cayetano', 'pia-cayetano'}),
    ('Compare senators', 'SINGLE_SUBJECT_SCOPE', set()),
])
def test_scope_switches_without_retrieval(senators, question, kind, targets):
    class NoCalls:
        def run(self, *a, **kw):
            pytest.fail('Routing must not call retrieval or generation')
    result = ChatbotService(senators, NoCalls()).ask(question, 'risa-hontiveros')
    assert result['type'] == kind
    assert {t['id'] for t in result['switch_targets']} == targets


def test_common_go_is_not_a_name(senators):
    assert scope_response('go into more detail', senators.get('risa-hontiveros'), senators.senators) is None


def test_injected_wrong_congress_never_reaches_generation(senators):
    from ai.pipeline import RAGPipeline
    class Retriever:
        def retrieve(self, *a, **kw):
            assert kw['congress_id'] == '20'
            return [{**chunk('old', 'Education', ['risa-hontiveros']), 'congress_id': '19'}]
    class Generator:
        def generate(self, *a):
            pytest.fail('Wrong Congress reached generation')
    result = ChatbotService(senators, RAGPipeline(Retriever(), Generator())).ask('education', 'risa-hontiveros', congress_id='20')
    assert result['retrieval_count'] == 0


def test_shared_source_preserves_each_marker_and_full_locator():
    one = {**chunk('one', 'Education', ['alpha']), 'evidence_id': 'E1'}
    two = copy.deepcopy(one)
    two['evidence_id'] = 'E2'
    two['citation_sources'][0]['locator'] = 'p. 42, ' + 'precise section ' * 30
    citations = format_citations([one, two], ['E1', 'E2'])
    assert [c['evidence_id'] for c in citations] == ['E1', 'E2']
    assert citations[1]['locator'] == two['citation_sources'][0]['locator']


@pytest.fixture
def reviewed(tmp_path, cloud):
    chunks, decisions = [], []
    for senator in ['alpha', 'beta']:
        source = tmp_path / f'{senator}.md'
        source.write_text('Reviewed local fixture', encoding='utf-8')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        record = {**chunk(senator, 'Education record', [senator]), 'senator_id': senator,
                  'source_file': source.name, 'source_mime_type': 'text/markdown',
                  'source_sha256': digest, 'claim_id': f'CLM-{senator}', 'extraction_version': 'test-1'}
        chunks.append(record)
        decisions.append({'senator_id': senator, 'claim_id': record['claim_id'], 'decision': 'approved',
            'reviewer': 'Test reviewer (synthetic)', 'reviewed_at': '2026-11-26', 'source_sha256': digest,
            'congress_id': '20', 'period_basis': 'Test fixture period',
            'sources': [{'source_url': record['citation_sources'][0]['source_url'], 'locator': 'p. 1'}]})
    scope = [{'senator_id': senator, 'bloc': bloc, 'reviewer': 'Synthetic test reviewer',
              'reviewed_at': '2026-11-26', 'as_of': '2026-11-26', 'source': chunks[i]['citation_sources'][0]}
             for i, (senator, bloc) in enumerate([('alpha', 'majority'), ('beta', 'minority')])]
    review = {'release_id': cloud.release, 'collection': cloud.collection, 'reviewed_at': '2026-11-26', 'subjects': ['alpha', 'beta'], 'scope_review': scope, 'decisions': decisions}
    return chunks, review, tmp_path


def test_release_requires_human_approval_and_exact_source_bytes(reviewed, cloud):
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    assert manifest['expected_counts'] == {'alpha': 1, 'beta': 1}
    assert len(validate_package(selected, manifest, cloud)) == 2
    (root / 'alpha.md').write_text('Edited claim', encoding='utf-8')
    with pytest.raises(ValueError, match='bytes'):
        prepare_release(chunks, review, root)


def test_unapproved_and_candidate_claims_cannot_publish(reviewed):
    chunks, review, root = reviewed
    review['decisions'] = []
    with pytest.raises(ValueError, match='No human-approved'):
        prepare_release(chunks, review, root)


def test_release_checks_provenance_and_cloud_sizes(reviewed):
    chunks, review, root = reviewed
    chunks[0]['citation_sources'][0]['locator'] = 'unchecked locator'
    with pytest.raises(ValueError, match='exact locator'):
        prepare_release(chunks, review, root)
    rows = _prepare_chunks([chunk('large', 'x' * 17000, ['alpha'])])
    with pytest.raises(ValueError, match='document'):
        preflight_rows(rows)


def test_publication_refuses_active_collection_before_client_call(reviewed, cloud, monkeypatch):
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    monkeypatch.setenv('SENTO_ACTIVE_COLLECTION', cloud.collection)
    with pytest.raises(ValueError, match='never overwrite'):
        publish_release(_prepare_chunks(selected), manifest, cloud, client=object())


def test_cloud_record_scan_is_paginated():
    rows = _prepare_chunks([chunk(str(i), 'Education', ['alpha']) for i in range(601)])
    from ai.vector_store import _metadata_for
    offsets = []
    def get(limit, offset, include):
        offsets.append(offset)
        page = rows[offset:offset + limit]
        return {'ids': [r['index_id'] for r in page], 'metadatas': [_metadata_for(r) for r in page]}
    verify_collection(NS(get=get, count=lambda: 601), rows)
    assert offsets == [0, 300, 600]


def test_release_publication_and_runtime_schema_roundtrip(reviewed, cloud, tmp_path):
    import chromadb
    from chromadb.config import Settings
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    rows = validate_package(selected, manifest, cloud)
    client = chromadb.PersistentClient(path=str(tmp_path / 'publication'), settings=Settings(anonymized_telemetry=False))
    embedder = NS(provider_name='openai', model_name=cloud.embedding_model, embed=lambda texts: [[1.] + [0.] * 1535 for _ in texts])
    result = publish_release(rows, manifest, cloud, client=client, embedder=embedder)
    assert result['indexed'] == 2
    store = ChromaVectorStore(embedder, client=client, collection_name=cloud.collection, read_only=True, expected_metadata=cloud.metadata())
    assert store.retrieve('education', 'alpha', congress_id='20')[0]['senator_id'] == 'alpha'
    assert store.retrieve('education', 'alpha', congress_id='19') == []
    with pytest.raises(Exception):
        publish_release(rows, manifest, cloud, client=client, embedder=embedder)
    assert store.count() == 2


def test_failed_publication_stays_inactive(reviewed, cloud):
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    collection = fake_collection(cloud, publication_status='building')
    def fail(**kwargs):
        raise RuntimeError('simulated outage')
    collection.add = fail
    collection.modify = lambda **kw: pytest.fail('Failed upload must not become ready')
    client = NS(create_collection=lambda **kw: collection)
    with pytest.raises(RuntimeError):
        publish_release(_prepare_chunks(selected), manifest, cloud, client=client, embedder=FakeEmbedder())
    assert collection.metadata['publication_status'] == 'building'


def test_cli_dry_run_never_connects_or_embeds(reviewed, cloud, monkeypatch, tmp_path, capsys):
    from tools.index_corpus import main
    from ai.releases import checksum
    import sys
    chunks, review, root = reviewed
    selected, manifest = prepare_release(chunks, review, root)
    chunks_path = tmp_path / 'chunks.jsonl'
    chunks_path.write_text(''.join(json.dumps(c) + '\n' for c in selected), encoding='utf-8')
    review_path = tmp_path / 'review.json'
    review_path.write_text(json.dumps(review), encoding='utf-8')
    manifest.update(chunks_sha256=checksum(chunks_path), review_sha256=checksum(review_path))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr(RAGConfig, 'cloud_client', lambda self: pytest.fail('Offline check connected'))
    monkeypatch.setattr(RAGConfig, 'embedder', lambda self: pytest.fail('Offline check constructed provider'))
    monkeypatch.delenv('OPENAI_API_KEY')
    monkeypatch.delenv('CHROMA_API_KEY')
    monkeypatch.setattr(sys, 'argv', ['index_corpus.py', '--chunks', str(chunks_path), '--release-manifest', str(manifest_path), '--dry-run'])
    assert main() == 0
    assert 'offline_preflight_passed' in capsys.readouterr().out
    chunks_path.write_text('{}\n', encoding='utf-8')
    assert main() == 2
