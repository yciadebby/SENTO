from collections.abc import Callable
import logging

from backend.senator_service import SenatorService
from ai.rag_factory import build_default_pipeline
from ai.rag_config import RAGConfigurationError
from backend.chat_scope import scope_response
from backend.security import normalize_question, requests_internal_prompt, requests_bulk_retrieval
from backend.policies import policy_violation


class ChatbotService:
    def __init__(self, senators=None, pipeline=None, *, settings=None, require_cloud=False):
        self.senators = senators or SenatorService()
        self.configuration_error = False
        try:
            self.pipeline = pipeline or build_default_pipeline(
                settings=settings,
                require_cloud=require_cloud,
            )
        except ValueError as exc:
            self.pipeline = None
            self.configuration_error = True
            # Only our controlled validation messages are safe for deployment logs.
            # Never log arbitrary provider exceptions or the supplied settings.
            detail = str(exc) if isinstance(exc, RAGConfigurationError) else 'Pipeline configuration could not be initialized.'
            logging.getLogger(__name__).warning('SENTO configuration error: %s', detail)

    def ask(self, question, senator_id, congress_id=None, progress_callback: Callable[[str], None] | None = None):
        question = normalize_question(question)
        senator = self.senators.get(senator_id)
        guarded = policy_violation(question) or requests_internal_prompt(question) or requests_bulk_retrieval(question)
        scoped = None if guarded else scope_response(question, senator, self.senators.senators)
        if scoped:
            return scoped
        if self.configuration_error:
            return {'answer': 'The answer service is unavailable because deployment settings are incomplete or inconsistent.',
                    'label': 'RAG CONFIGURATION ERROR', 'topic': 'sources', 'citations': [], 'retrieval_count': 0,
                    'limitation': 'The maintainer must check deployment settings. No model answer was generated.',
                    'validation': {'status': 'CONFIGURATION_ERROR', 'verified': False, 'citation_ids': []}}
        run_options = {}
        if congress_id:
            run_options["congress_id"] = congress_id
        if progress_callback:
            run_options["progress_callback"] = progress_callback
        return self.pipeline.run(question, senator, self.senators.style(senator_id), **run_options)

    def coverage(self, senator_id):
        if self.configuration_error:
            return {'status': 'configuration_error', 'count': None, 'reviewed_at': None}
        retriever = self.pipeline.retriever
        config = getattr(retriever, 'config', None)
        if not config or config.backend == 'local':
            return {'status': 'local_development', 'count': None, 'reviewed_at': None}
        try:
            store = retriever._get_store()
            metadata = store.metadata
            import json
            counts = metadata.get('expected_counts') if config.backend == 'qdrant_cloud' else json.loads(metadata.get('coverage_json', '{}'))
            return {'status': 'project_curated_release', 'count': counts.get(senator_id, 0),
                    'curation_status': metadata.get('curation_status'), 'release_id': config.release}
        except Exception:
            return {'status': 'service_unavailable', 'count': None, 'reviewed_at': None}
    def indexed_passage_count(self):
        if self.pipeline is None:
            return None
        count_indexed = getattr(self.pipeline.retriever, "count_indexed", None)
        if not callable(count_indexed):
            return None
        try:
            return count_indexed()
        except Exception:
            return None
