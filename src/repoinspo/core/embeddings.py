"""Embedding helpers for reranking repository candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import faiss
import numpy as np
from litellm import aembedding

ItemT = TypeVar("ItemT")


class EmbeddingIndex:
    """FAISS-backed embedding index for hybrid repository search."""

    def __init__(
        self,
        model: str,
        embedding_func: Any = aembedding,
        persist_path: Path | None = None,
    ) -> None:
        self.model = model
        self.embedding_func = embedding_func
        self.persist_path = persist_path or Path.home() / ".repoinspo" / "index.faiss"
        self._index: faiss.IndexFlatIP | None = None

    async def rerank(
        self,
        query: str,
        items: list[ItemT],
        texts: list[str],
        top_k: int,
    ) -> list[ItemT]:
        """Rank items by cosine similarity against a query text."""

        if not items:
            return []

        document_vectors = await self._embed(texts)
        matrix = np.asarray(document_vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

        query_vector = np.asarray(await self._embed([query]), dtype="float32")
        faiss.normalize_L2(query_vector)
        _, indices = self._index.search(query_vector, min(top_k, len(items)))
        return [items[index] for index in indices[0] if index != -1]

    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("Cannot persist an empty embedding index")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.persist_path))

    def load(self) -> None:
        self._index = faiss.read_index(str(self.persist_path))

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        response = await self.embedding_func(model=self.model, input=inputs)
        data = response["data"] if isinstance(response, dict) else response.data
        return [item["embedding"] if isinstance(item, dict) else item.embedding for item in data]
