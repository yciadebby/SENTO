"""Validated deployment settings. Never loads .env or includes secrets in repr."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


class RAGConfigurationError(ValueError):
    """Maintainer-safe validation message: never include supplied values or credentials."""


@dataclass(frozen=True)
class RAGConfig:
    backend: str
    collection: str
    release: str
    embedding_provider: str
    embedding_model: str
    dimensions: int
    embedding_base_url: str
    tenant: str
    database: str
    host: str
    path: str
    top_k: int
    max_distance: float
    qdrant_url: str = ''
    generator_provider: str = 'ollama'
    openai_model: str = 'gpt-6-luna'
    openai_reasoning_effort: str = 'low'
    openai_base_url: str = 'https://api.openai.com/v1'
    qdrant_api_key: str = field(default='', repr=False, compare=False)
    openai_api_key: str = field(default='', repr=False, compare=False)
    fireworks_api_key: str = field(default='', repr=False, compare=False)
    chroma_api_key: str = field(default='', repr=False, compare=False)
    global_request_limit: int = 30
    global_request_window_seconds: float = 60.0

    @classmethod
    def load(cls, *, require_credentials=True, require_cloud=False, settings=None):
        """Load validated settings from a mapping or the process environment.

        The Streamlit app passes ``st.secrets`` as a mapping. Maintenance CLIs
        may continue to use environment variables and their explicitly loaded
        private TOML file. Secrets are kept on the config object with repr off.
        """
        if require_cloud and settings is None:
            raise RAGConfigurationError('The hosted SEN’TO chat runtime requires Streamlit Secrets.')
        if settings is None:
            env = os.environ
            qdrant_api_key = env.get('QDRANT_API_KEY', '')
        else:
            if not isinstance(settings, Mapping):
                raise RAGConfigurationError('Streamlit Secrets must be a root-level TOML mapping.')
            env = {}
            for key, value in settings.items():
                if not key.startswith(('SENTO_', 'QDRANT_', 'CHROMA_', 'OPENAI_', 'FIREWORKS_')):
                    continue
                if not isinstance(value, (str, int, float, bool)):
                    raise RAGConfigurationError('Provider settings must be root-level scalar Streamlit Secrets.')
                env[key] = str(value)
            # The public app intentionally accepts only the separately named
            # read-only key. It never reads the publisher's QDRANT_API_KEY.
            qdrant_api_key = env.get('SENTO_QDRANT_READ_ONLY_API_KEY', '')
        backend = env.get('SENTO_VECTOR_BACKEND', 'local')
        if backend not in {'local', 'chroma_cloud', 'qdrant_cloud'}:
            raise RAGConfigurationError('SENTO_VECTOR_BACKEND must be local, chroma_cloud or qdrant_cloud.')
        if require_cloud and backend != 'qdrant_cloud':
            raise RAGConfigurationError('The hosted SEN’TO chat runtime requires Qdrant Cloud.')
        cloud = backend != 'local'
        qdrant = backend == 'qdrant_cloud'
        collection_key = 'SENTO_QDRANT_COLLECTION' if qdrant else 'SENTO_CHROMA_COLLECTION'
        provider = env.get('SENTO_EMBEDDING_PROVIDER', 'openai' if cloud else 'ollama')
        if provider not in {'openai', 'fireworks', 'ollama', 'local_hash'}:
            raise RAGConfigurationError('SENTO_EMBEDDING_PROVIDER must be openai, fireworks, local_hash or ollama.')
        default_model = (
            'text-embedding-3-small'
            if provider == 'openai'
            else ('sento-hash-ngram-v1' if provider == 'local_hash' else env.get('SENTO_OLLAMA_EMBED_MODEL', 'embeddinggemma'))
        )
        if provider == 'fireworks':
            default_model = 'accounts/fireworks/models/qwen3-embedding-8b'
        model = env.get('SENTO_EMBEDDING_MODEL', default_model)
        default_dimensions = '4096' if provider == 'fireworks' else ('1536' if provider in {'openai', 'local_hash'} else '0')
        try:
            dimensions = int(env.get('SENTO_EMBEDDING_DIMENSIONS', default_dimensions))
            top_k = int(env.get('SENTO_RETRIEVAL_TOP_K', '5'))
            distance = float(env.get('SENTO_MAX_COSINE_DISTANCE', '.65'))
            global_request_limit = int(env.get('SENTO_GLOBAL_REQUEST_LIMIT', '30'))
            global_request_window = float(env.get('SENTO_GLOBAL_REQUEST_WINDOW_SECONDS', '60'))
        except ValueError:
            raise RAGConfigurationError('Embedding, retrieval and request-limit settings must be numeric.') from None
        if not 1 <= top_k <= 5 or not math.isfinite(distance) or not 0 <= distance <= 2:
            raise RAGConfigurationError('Top K must be 1–5 and cosine distance must be 0–2.')
        if not 1 <= global_request_limit <= 1000 or not math.isfinite(global_request_window) or not 1 <= global_request_window <= 3600:
            raise RAGConfigurationError('The shared request limit must be 1–1000 and its window 1–3600 seconds.')
        if provider == 'openai' and (model != 'text-embedding-3-small' or dimensions != 1536):
            raise RAGConfigurationError('This release requires text-embedding-3-small with 1536 dimensions.')
        if provider == 'local_hash' and (model != 'sento-hash-ngram-v1' or dimensions != 1536):
            raise RAGConfigurationError('local_hash embeddings require sento-hash-ngram-v1 with 1536 dimensions.')
        if provider == 'fireworks' and (model != 'accounts/fireworks/models/qwen3-embedding-8b' or dimensions != 4096):
            raise RAGConfigurationError('Fireworks embeddings require accounts/fireworks/models/qwen3-embedding-8b with 4096 dimensions.')
        default_embedding_url = 'https://api.fireworks.ai/inference/v1' if provider == 'fireworks' else 'https://api.openai.com/v1'
        embedding_url = env.get('SENTO_EMBEDDING_BASE_URL', default_embedding_url)
        openai_base_url = env.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        if provider in {'openai', 'fireworks'}:
            parsed = urlsplit(embedding_url)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise RAGConfigurationError('Hosted embeddings require an explicit HTTPS base URL without credentials or query parameters.')
        parsed_generation_url = urlsplit(openai_base_url)
        if (parsed_generation_url.scheme != 'https' or not parsed_generation_url.hostname
                or parsed_generation_url.username or parsed_generation_url.password
                or parsed_generation_url.query or parsed_generation_url.fragment):
            raise RAGConfigurationError('OpenAI generation requires an HTTPS base URL without credentials or query parameters.')
        generator_provider = env.get('SENTO_GENERATOR_PROVIDER', 'ollama').strip().casefold()
        if generator_provider not in {'openai', 'ollama'}:
            raise RAGConfigurationError("SENTO_GENERATOR_PROVIDER must be 'ollama' or 'openai'.")
        openai_model = env.get('SENTO_OPENAI_MODEL', 'gpt-6-luna').strip()
        reasoning_effort = env.get('SENTO_OPENAI_REASONING_EFFORT', 'low').strip().casefold()
        if not openai_model:
            raise RAGConfigurationError('SENTO_OPENAI_MODEL must name a generation model.')
        if reasoning_effort not in {'none', 'low', 'medium', 'high', 'xhigh', 'max'}:
            raise RAGConfigurationError('SENTO_OPENAI_REASONING_EFFORT is not supported.')
        if cloud:
            required = [collection_key, 'SENTO_RELEASE_ID']
            required += ['SENTO_QDRANT_URL'] if qdrant else ['CHROMA_TENANT', 'CHROMA_DATABASE']
            if require_credentials:
                qdrant_credential = ('SENTO_QDRANT_READ_ONLY_API_KEY' if settings is not None
                                     else 'QDRANT_API_KEY')
                required += [qdrant_credential if qdrant else 'CHROMA_API_KEY', 'OPENAI_API_KEY']
                if provider == 'fireworks':
                    required.append('FIREWORKS_API_KEY')
            missing = [key for key in required if not env.get(key, '').strip()]
            if missing:
                raise RAGConfigurationError('Cloud configuration is incomplete. Missing settings: ' + ', '.join(missing) + '.')
            if provider not in {'openai', 'fireworks', 'local_hash'} or generator_provider != 'openai':
                raise RAGConfigurationError('Cloud mode requires hosted OpenAI generation and non-Ollama embeddings; no local fallback is allowed.')
            if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{1,125}[a-zA-Z0-9]', env[collection_key]):
                raise RAGConfigurationError('Invalid cloud collection name.')
        qdrant_url = env.get('SENTO_QDRANT_URL', '').rstrip('/')
        if qdrant:
            try:
                parsed = urlsplit(qdrant_url)
                safe = (parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password
                        and not parsed.query and not parsed.fragment and not parsed.path and parsed.port in (None, 443, 6333))
            except ValueError:
                safe = False
            if not safe:
                raise RAGConfigurationError('SENTO_QDRANT_URL must be an HTTPS endpoint without credentials, path or query.')
        return cls(
            backend=backend,
            collection=env.get(collection_key, ''),
            release=env.get('SENTO_RELEASE_ID', ''),
            embedding_provider=provider,
            embedding_model=model,
            dimensions=dimensions,
            embedding_base_url=embedding_url,
            tenant=env.get('CHROMA_TENANT', ''),
            database=env.get('CHROMA_DATABASE', ''),
            host=env.get('CHROMA_HOST', 'api.trychroma.com'),
            path=env.get('SENTO_CHROMA_PATH', str(Path(__file__).resolve().parents[1] / 'data/corpus/processed/chroma')),
            top_k=top_k,
            max_distance=distance,
            qdrant_url=qdrant_url,
            generator_provider=generator_provider,
            openai_model=openai_model,
            openai_reasoning_effort=reasoning_effort,
            openai_base_url=openai_base_url,
            qdrant_api_key=qdrant_api_key,
            openai_api_key=env.get('OPENAI_API_KEY', ''),
            fireworks_api_key=env.get('FIREWORKS_API_KEY', ''),
            chroma_api_key=env.get('CHROMA_API_KEY', ''),
            global_request_limit=global_request_limit,
            global_request_window_seconds=global_request_window,
        )

    def embedder(self):
        from ai.embeddings import OpenAIEmbeddingProvider, FireworksEmbeddingProvider, OllamaEmbeddingProvider, LocalHashEmbeddingProvider
        if self.embedding_provider == 'openai':
            return OpenAIEmbeddingProvider(api_key=self.openai_api_key, base_url=self.embedding_base_url,
                                           model=self.embedding_model, dimensions=self.dimensions)
        if self.embedding_provider == 'fireworks':
            return FireworksEmbeddingProvider(api_key=self.fireworks_api_key, base_url=self.embedding_base_url,
                                             model=self.embedding_model, dimensions=self.dimensions)
        if self.embedding_provider == 'local_hash':
            return LocalHashEmbeddingProvider(model=self.embedding_model, dimensions=self.dimensions)
        return OllamaEmbeddingProvider(model=self.embedding_model)

    def cloud_client(self):
        if self.backend == 'qdrant_cloud':
            from qdrant_client import QdrantClient
            return QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key,
                                port=443 if urlsplit(self.qdrant_url).port is None else urlsplit(self.qdrant_url).port,
                                timeout=45, prefer_grpc=False, check_compatibility=False)
        import chromadb
        from chromadb.config import Settings
        return chromadb.CloudClient(tenant=self.tenant, database=self.database,
                                    api_key=self.chroma_api_key, cloud_host=self.host,
                                    settings=Settings(anonymized_telemetry=False))

    def metadata(self):
        return {'schema_version': '2', 'embedding_provider': self.embedding_provider,
                'embedding_model': self.embedding_model, 'embedding_dimensions': self.dimensions,
                'release_id': self.release, 'distance_metric': 'cosine'}
