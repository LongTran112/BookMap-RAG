"""Shared retrieval filter type used by both search and RAG services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class BookFilters:
    categories: Optional[Sequence[str]] = None
    learning_modes: Optional[Sequence[str]] = None
    min_similarity: float = -1.0
