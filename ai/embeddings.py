"""Embedding provider interfaces and the local Ollama adapter."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Protocol
from threading import RLock
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    """Embedding provider failed or returned invalid vectors."""


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HostedEmbeddingProvider:
    """Explicit hosted embeddings, with bounded requests and strict response validation."""
    def __init__(self, *, api_key, base_url, model, dimensions, client=None):
        self._model, self.dimensions = model, dimensions
        self.client = client
        self._client_lock = RLock()
        self._api_key, self.base_url = api_key, base_url

    @property
    def model_name(self):
        return self._model

    def embed(self, texts):
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must be nonempty strings.")
        try:
            with self._client_lock:
                if self.client is None:
                    if not self._api_key:
                        raise EmbeddingError("Hosted embedding credentials are missing.")
                    from openai import OpenAI
                    self.client = OpenAI(api_key=self._api_key, base_url=self.base_url, timeout=45, max_retries=0)
            response = self.client.embeddings.create(model=self.model_name, input=texts, dimensions=self.dimensions)
            rows = response.data
            if len(rows) != len(texts) or sorted(row.index for row in rows) != list(range(len(texts))):
                raise EmbeddingError("Unexpected embedding count or indexes.")
            vectors = []
            for row in sorted(rows, key=lambda row: row.index):
                vector = row.embedding
                if len(vector) != self.dimensions or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                    raise EmbeddingError("Malformed embedding dimensions or numeric values.")
                vectors.append([float(v) for v in vector])
            return vectors
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingError("The hosted embedding service is unavailable.") from None


class OpenAIEmbeddingProvider(HostedEmbeddingProvider):
    provider_name = "openai"

    def __init__(self, *, api_key, base_url, model="text-embedding-3-small", dimensions=1536, client=None):
        if model != "text-embedding-3-small" or dimensions != 1536:
            raise ValueError("Hosted embeddings require text-embedding-3-small / 1536.")
        super().__init__(api_key=api_key, base_url=base_url, model=model, dimensions=dimensions, client=client)


class FireworksEmbeddingProvider(HostedEmbeddingProvider):
    """Qwen3 embeddings using the existing OpenAI-compatible client, with a separate key."""
    provider_name = "fireworks"
    MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
    DIMENSIONS = 4096
    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(self, *, api_key, base_url=BASE_URL, model=MODEL, dimensions=DIMENSIONS, client=None):
        if model != self.MODEL or dimensions != self.DIMENSIONS:
            raise ValueError("Fireworks embeddings require accounts/fireworks/models/qwen3-embedding-8b / 4096.")
        super().__init__(api_key=api_key, base_url=base_url, model=model, dimensions=dimensions, client=client)


class OllamaEmbeddingProvider:
    """Call Ollama's local /api/embed endpoint without coupling Chroma to Ollama."""

    provider_name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None, timeout: float = 60):
        self._model = model or os.environ.get("SENTO_OLLAMA_EMBED_MODEL", "embeddinggemma")
        self.base_url = (base_url or os.environ.get("SENTO_OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout
        if not self._model.strip():
            raise ValueError("SENTO_OLLAMA_EMBED_MODEL must name an installed Ollama embedding model.")

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be non-empty strings.")
        request_body = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/embed",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise EmbeddingError(
                "Could not obtain embeddings from Ollama. Confirm it is running and the configured model is installed."
            ) from exc

        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected embedding count.")
        dimension = None
        normalized: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("Ollama returned an empty or malformed embedding.")
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError("Ollama returned a non-numeric embedding value.") from exc
            if not all(math.isfinite(value) for value in values):
                raise EmbeddingError("Ollama returned a non-finite embedding value.")
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise EmbeddingError("Ollama returned vectors with inconsistent dimensions.")
            normalized.append(values)
        return normalized


class LocalHashEmbeddingProvider:
    """Deterministic in-process feature-hashing embedder for zero-dependency cloud/local retrieval."""

    provider_name = "local_hash"

    def __init__(self, model: str = "sento-hash-ngram-v1", dimensions: int = 1536):
        if model != "sento-hash-ngram-v1" or dimensions != 1536:
            raise ValueError("LocalHashEmbeddingProvider requires sento-hash-ngram-v1 / 1536.")
        self._model = model
        self.dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    def _embed_one(self, text: str) -> list[float]:
        folded = "".join(
            ch for ch in unicodedata.normalize("NFKD", text.casefold())
            if not unicodedata.combining(ch)
        )
        words = re.findall(r"[a-z0-9]+", folded)
        vec = [0.0] * self.dimensions

        def _add_feature(feature: str, weight: float) -> None:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            idx = (raw >> 1) % self.dimensions
            sign = 1.0 if (raw & 1) else -1.0
            vec[idx] += sign * weight

        for word in words:
            _add_feature(f"w:{word}", 2.0)
            if len(word) >= 3:
                padded = f"^{word}$"
                for i in range(len(padded) - 2):
                    _add_feature(f"c3:{padded[i:i + 3]}", 0.7)
            if len(word) >= 4:
                for i in range(len(word) - 3):
                    _add_feature(f"c4:{word[i:i + 4]}", 0.8)
        for i in range(len(words) - 1):
            _add_feature(f"b:{words[i]}_{words[i + 1]}", 1.5)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            vec[0] = 1.0
            return vec
        inv = 1.0 / norm
        return [v * inv for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must be nonempty strings.")
        return [self._embed_one(text) for text in texts]
