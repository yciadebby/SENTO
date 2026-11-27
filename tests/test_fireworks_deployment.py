"""Offline regressions for the Fireworks / OpenAI / Qdrant deployment contract."""
from dataclasses import replace
from contextlib import closing
import json
import sys
from types import SimpleNamespace as NS

import pytest
from qdrant_client import QdrantClient

from ai.embeddings import EmbeddingError, FireworksEmbeddingProvider
from ai.qdrant_store import QdrantVectorStore
from ai.rag_config import RAGConfig
from ai.rag_factory import build_default_pipeline
from ai.releases import prepare_markdown_release, publish_release, validate_package
from ai.senator_markdown import ingest_project_curated_markdown
from ai.vector_store import VectorStoreError
from backend.chatbot_service import ChatbotService


@pytest.fixture
def settings():
    return {
        'SENTO_VECTOR_BACKEND': 'qdrant_cloud',
        'SENTO_QDRANT_URL': 'https://example.cloud.qdrant.io',
        'SENTO_QDRANT_COLLECTION': 'sento_demo_md_fireworks_qwen3_8b_4096',
        'SENTO_RELEASE_ID': 'demo-md-fireworks-qwen3-8b-r1',
        'SENTO_QDRANT_READ_ONLY_API_KEY': 'fake-read-key',
        'SENTO_EMBEDDING_PROVIDER': 'fireworks',
        'SENTO_EMBEDDING_MODEL': 'accounts/fireworks/models/qwen3-embedding-8b',
        'SENTO_EMBEDDING_DIMENSIONS': '4096',
        'SENTO_EMBEDDING_BASE_URL': 'https://api.fireworks.ai/inference/v1',
        'FIREWORKS_API_KEY': 'fake-fireworks-key',
        'SENTO_GENERATOR_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'fake-openai-key',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1',
        'SENTO_OPENAI_MODEL': 'gpt-6-luna',
        'SENTO_OPENAI_REASONING_EFFORT': 'low',
    }


def test_streamlit_secrets_construct_separate_clients_and_rotate_cache(settings, monkeypatch):
    import frontend.api as api
    import openai
    monkeypatch.setattr(api.st, 'secrets', settings)
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: pytest.fail('No startup network client'))
    captured = []
    monkeypatch.setattr(api, '_cached_chatbot_service', lambda key, values: captured.append((key, values)))
    api.get_chatbot_service()
    settings['FIREWORKS_API_KEY'] = 'rotated-fireworks-key'
    api.get_chatbot_service()
    assert captured[0][0] != captured[1][0]
    pipeline = build_default_pipeline(settings=captured[1][1], require_cloud=True)
    embedder = pipeline.retriever.embedder
    assert isinstance(embedder, FireworksEmbeddingProvider)
    assert embedder._api_key == 'rotated-fireworks-key'
    assert embedder.base_url == settings['SENTO_EMBEDDING_BASE_URL']
    assert pipeline.generator.provider._api_key == 'fake-openai-key'
    assert pipeline.generator.provider.base_url == settings['OPENAI_BASE_URL']
    assert pipeline.retriever.config.qdrant_api_key == 'fake-read-key'
    assert 'rotated-fireworks-key' not in repr(pipeline.retriever.config)


@pytest.mark.parametrize('key', ['FIREWORKS_API_KEY', 'OPENAI_API_KEY', 'SENTO_QDRANT_READ_ONLY_API_KEY'])
def test_missing_keys_fail_closed_without_environment_fallback(settings, monkeypatch, caplog, key):
    del settings[key]
    monkeypatch.setenv(key, 'environment-key-must-not-be-used')
    settings['QDRANT_API_KEY'] = 'publisher-key-must-not-be-used'
    bot = ChatbotService(settings=settings, require_cloud=True)
    response = bot.ask('What records are available?', 'risa-hontiveros')
    assert response['validation']['status'] == 'CONFIGURATION_ERROR'
    assert key in caplog.text
    assert all(secret not in caplog.text for secret in ['fake-openai-key', 'fake-fireworks-key',
                                                       'publisher-key-must-not-be-used', 'environment-key-must-not-be-used'])
    assert 'API_KEY' not in response['answer']


def test_arbitrary_pipeline_errors_are_not_logged(settings, monkeypatch, caplog):
    def broken(**kwargs):
        raise ValueError('private-provider-response-and-key')
    monkeypatch.setattr('backend.chatbot_service.build_default_pipeline', broken)
    assert ChatbotService(settings=settings, require_cloud=True).configuration_error
    assert 'private-provider-response-and-key' not in caplog.text


@pytest.mark.parametrize('changes', [
    {'SENTO_EMBEDDING_DIMENSIONS': '1536'},
    {'SENTO_EMBEDDING_MODEL': 'text-embedding-3-small'},
    {'SENTO_EMBEDDING_BASE_URL': 'http://api.fireworks.ai/inference/v1'},
    {'SENTO_EMBEDDING_BASE_URL': 'https://secret@example.com/v1'},
])
def test_incompatible_fireworks_settings_rejected(settings, changes):
    with pytest.raises(ValueError):
        RAGConfig.load(settings={**settings, **changes}, require_cloud=True)


def test_fireworks_requests_use_4096_dimensions_and_only_fireworks_key(settings, monkeypatch):
    import openai
    clients, requests = [], []
    def create(**kwargs):
        requests.append(kwargs)
        return NS(data=[NS(index=1, embedding=[2.] * 4096), NS(index=0, embedding=[1.] * 4096)])
    def client(**kwargs):
        clients.append(kwargs)
        return NS(embeddings=NS(create=create))
    monkeypatch.setattr(openai, 'OpenAI', client)
    embedder = RAGConfig.load(settings=settings, require_cloud=True).embedder()
    assert [v[0] for v in embedder.embed(['first', 'second'])] == [1., 2.]
    assert clients == [{'api_key': 'fake-fireworks-key', 'base_url': settings['SENTO_EMBEDDING_BASE_URL'],
                        'timeout': 45, 'max_retries': 0}]
    assert requests == [{'model': settings['SENTO_EMBEDDING_MODEL'], 'input': ['first', 'second'], 'dimensions': 4096}]


@pytest.mark.parametrize('rows', [[], [NS(index=1, embedding=[1.] * 4096)],
    [NS(index=0, embedding=[1.] * 1536)], [NS(index=0, embedding=[float('inf')] * 4096)],
    [NS(index=0, embedding=[True] * 4096)], [NS(index=0, embedding=['1'] * 4096)]])
def test_fireworks_malformed_vectors_rejected(rows):
    embedder = FireworksEmbeddingProvider(api_key='fake-key',
        client=NS(embeddings=NS(create=lambda **kw: NS(data=rows))))
    with pytest.raises(EmbeddingError):
        embedder.embed(['text'])


def test_fireworks_provider_exception_is_redacted():
    def broken(**kw):
        raise RuntimeError('private-fireworks-key')
    embedder = FireworksEmbeddingProvider(api_key='fake-key', client=NS(embeddings=NS(create=broken)))
    with pytest.raises(EmbeddingError) as exc:
        embedder.embed(['text'])
    assert 'private-fireworks-key' not in str(exc.value)


@pytest.mark.filterwarnings('ignore:Payload indexes have no effect in the local Qdrant')
def test_fireworks_release_roundtrip_and_identity_checks(settings, tmp_path, monkeypatch):
    for subject in ['risa-hontiveros', 'raffy-tulfo']:
        (tmp_path / f'{subject}.md').write_text(
            f'---\nsenator_id: {subject}\n---\n## Synthetic fixture\n'
            'Synthetic test passage only.\nCongress: 20\n', encoding='utf-8')
    config = RAGConfig.load(settings=settings, require_cloud=True)
    chunks = ingest_project_curated_markdown(tmp_path)
    selected, manifest = prepare_markdown_release(chunks, config.release, config.collection, tmp_path,
                                                 embedding_provider='fireworks')
    assert manifest['embedding_dimensions'] == 4096
    rows = validate_package(selected, manifest, config)
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_package(selected, manifest, replace(config, embedding_provider='openai'))
    embedder = config.embedder()
    embedder.client = NS(embeddings=NS(create=lambda **kw: NS(data=[
        NS(index=i, embedding=[1.] + [0.] * 4095) for i in range(len(kw['input']))])))
    monkeypatch.setenv('SENTO_ACTIVE_COLLECTION', 'NONE')
    with closing(QdrantClient(':memory:')) as client:
        publish_release(rows, manifest, config, client=client, embedder=embedder)
        store = QdrantVectorStore(embedder, config, client=client)
        hits = store.retrieve('Synthetic fixture', 'risa-hontiveros', congress_id='20')
        assert len(hits) == 1 and hits[0]['senator_id'] == 'risa-hontiveros'
        assert hits[0]['citation_sources']
        assert store.retrieve('Synthetic fixture', 'risa-hontiveros', congress_id='19') == []
        with pytest.raises(VectorStoreError, match='identity'):
            QdrantVectorStore(embedder, replace(config, release='wrong-release'), client=client)
        with pytest.raises(VectorStoreError, match='dimensions'):
            QdrantVectorStore(embedder, replace(config, dimensions=1536), client=client)


def test_private_cli_settings_and_offline_runtime_check(settings, tmp_path, monkeypatch, capsys):
    from tools.runtime_settings import load_settings
    from tools.check_rag import main
    private = tmp_path / 'private.toml'
    private.write_text('\n'.join(f'{key} = {json.dumps(value)}' for key, value in settings.items()), encoding='utf-8')
    monkeypatch.delenv('FIREWORKS_API_KEY', raising=False)
    monkeypatch.setattr(sys, 'argv', ['check_rag', '--secrets-file', str(private), '--runtime-settings', '--config-only'])
    monkeypatch.setattr(RAGConfig, 'cloud_client', lambda self: pytest.fail('Offline check connected'))
    monkeypatch.setattr(RAGConfig, 'embedder', lambda self: pytest.fail('Offline check built embedder'))
    assert main() == 0
    assert 'configuration_ok' in capsys.readouterr().out
    import os
    assert 'FIREWORKS_API_KEY' not in os.environ
    # The maintenance loader still exports the embedding key for publishing.
    for key in settings:
        monkeypatch.setenv(key, '')
    load_settings(private)
    assert os.environ['FIREWORKS_API_KEY'] == 'fake-fireworks-key'
    del settings['SENTO_QDRANT_READ_ONLY_API_KEY']
    settings['QDRANT_API_KEY'] = 'fake-publisher-key'
    private.write_text('\n'.join(f'{key} = {json.dumps(value)}' for key, value in settings.items()), encoding='utf-8')
    assert main() == 2
    output = capsys.readouterr()
    assert 'SENTO_QDRANT_READ_ONLY_API_KEY' in output.err
    assert 'fake-publisher-key' not in output.err
