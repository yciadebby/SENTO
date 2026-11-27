"""Frontend-facing service seams.

These functions keep the UI independent from transport and backend details.
Replace their bodies with API calls when a backend endpoint is ready.
"""

def get_senator_directory():
    """Load the local example directory until a directory endpoint exists."""
    # TODO: Replace with GET /api/senators when the backend endpoint exists.
    from backend.senator_service import SenatorService

    return SenatorService()


import hashlib
import json
from copy import deepcopy
from time import monotonic
import streamlit as st


_CHAT_SECRET_NAMES = (
    'SENTO_VECTOR_BACKEND',
    'SENTO_QDRANT_URL',
    'SENTO_QDRANT_COLLECTION',
    'SENTO_RELEASE_ID',
    'SENTO_QDRANT_READ_ONLY_API_KEY',
    'SENTO_EMBEDDING_PROVIDER',
    'SENTO_EMBEDDING_MODEL',
    'SENTO_EMBEDDING_DIMENSIONS',
    'SENTO_EMBEDDING_BASE_URL',
    'FIREWORKS_API_KEY',
    'SENTO_GENERATOR_PROVIDER',
    'OPENAI_API_KEY',
    'OPENAI_BASE_URL',
    'SENTO_OPENAI_MODEL',
    'SENTO_OPENAI_REASONING_EFFORT',
    'SENTO_RETRIEVAL_TOP_K',
    'SENTO_MAX_COSINE_DISTANCE',
    'SENTO_GLOBAL_REQUEST_LIMIT',
    'SENTO_GLOBAL_REQUEST_WINDOW_SECONDS',
)

SESSION_RESPONSE_CACHE_KEY = '_sento_chat_response_cache'
SESSION_RESPONSE_CACHE_TTL_SECONDS = 600
SESSION_RESPONSE_CACHE_MAX_ENTRIES = 32
SESSION_RESPONSE_CACHE_VERSION = '1'


def _chat_runtime_secrets():
    """Read only root-level RAG settings from Streamlit Secrets."""
    try:
        secrets = st.secrets
        settings = {}
        for name in _CHAT_SECRET_NAMES:
            try:
                if name not in secrets:
                    continue
                value = secrets[name]
            except Exception:
                continue
            if isinstance(value, (str, int, float, bool)):
                settings[name] = value
        return settings
    except Exception:
        # Missing Secrets is reported by ChatbotService as a configuration error.
        return {}


@st.cache_resource(show_spinner=False)
def _cached_chatbot_service(settings_key, _settings):
    """Build the cloud-only RAG service for the current secret version."""
    from backend.chatbot_service import ChatbotService

    return ChatbotService(settings=_settings, require_cloud=True)


def get_chatbot_service():
    settings = _chat_runtime_secrets()
    # Hash credentials so rotations rebuild clients without putting secrets in
    # Streamlit's cache key or passing them through the process environment.
    payload = json.dumps(settings, sort_keys=True, separators=(',', ':'), default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return _cached_chatbot_service(digest, settings)


def _session_response_cache_key(service, senator_id, message, congress_id):
    pipeline = getattr(service, 'pipeline', None)
    if pipeline is None:
        return None
    config = getattr(getattr(pipeline, 'retriever', None), 'config', None)
    if config is None:
        return None
    provider = getattr(getattr(pipeline, 'generator', None), 'provider', None)
    from ai.generation import SYSTEM_INSTRUCTIONS

    try:
        style = service.senators.style(senator_id)
    except Exception:
        return None
    settings_digest = hashlib.sha256(
        json.dumps(_chat_runtime_secrets(), sort_keys=True, separators=(',', ':'), default=str).encode()
    ).hexdigest()
    identity = {
        'cache_version': SESSION_RESPONSE_CACHE_VERSION,
        'runtime_settings': settings_digest,
        'question': message,
        'senator_id': senator_id,
        'congress_id': congress_id,
        'style': style,
        'release': config.release,
        'collection': config.collection,
        'embedding_model': config.embedding_model,
        'embedding_dimensions': config.dimensions,
        'embedding_base_url': config.embedding_base_url,
        'generation_provider': getattr(provider, 'provider_name', ''),
        'generation_model': getattr(provider, 'model', ''),
        'reasoning_effort': getattr(provider, 'reasoning_effort', ''),
        'generation_base_url': getattr(provider, 'base_url', ''),
        'top_k': config.top_k,
        'max_distance': config.max_distance,
        'instructions_sha256': hashlib.sha256(SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def send_chat_message(senator_id, message, congress_id=None, progress_callback=None):
    """Run one question through RAG, reusing exact recent answers in-session."""
    from backend.security import normalize_question

    message = normalize_question(message)
    service = get_chatbot_service()
    try:
        session_state = st.session_state
        cache = session_state.get(SESSION_RESPONSE_CACHE_KEY)
        if not isinstance(cache, dict):
            cache = {}
            session_state[SESSION_RESPONSE_CACHE_KEY] = cache
    except Exception:
        cache = None

    now = monotonic()
    if cache is not None:
        for key, entry in list(cache.items()):
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], (int, float))
                or now - entry[0] >= SESSION_RESPONSE_CACHE_TTL_SECONDS
            ):
                cache.pop(key, None)

    cache_key = _session_response_cache_key(service, senator_id, message, congress_id) if cache is not None else None
    if cache_key is not None and cache_key in cache:
        return deepcopy(cache[cache_key][1])

    options = {}
    if congress_id:
        options["congress_id"] = congress_id
    if progress_callback:
        options["progress_callback"] = progress_callback
    response = service.ask(message, senator_id, **options)
    validation = response.get('validation') if isinstance(response, dict) else None
    status = validation.get('status') if isinstance(validation, dict) else None
    if cache_key is not None and status not in {'SERVICE_UNAVAILABLE', 'CONFIGURATION_ERROR', 'RATE_LIMITED'}:
        cache[cache_key] = (now, deepcopy(response))
        while len(cache) > SESSION_RESPONSE_CACHE_MAX_ENTRIES:
            oldest_key = min(cache, key=lambda item: cache[item][0])
            cache.pop(oldest_key, None)
    return response


def get_comparison_records(directory, senator_ids, period="20th_congress"):
    """Read source-linked comparison claims from the curated Markdown profiles."""
    from backend.comparison_service import ComparisonService

    return ComparisonService(senators=directory).compare(senator_ids, period=period)
