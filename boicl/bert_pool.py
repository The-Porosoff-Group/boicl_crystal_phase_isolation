"""
bert_pool.py
============
Lightweight Pool for use with BertAskTellFewShotTopk.

Drops all Ada / FAISS / OpenAI embedding machinery from pool.py — those are
unused in the BERT workflow because approx_sample is replaced by
BertAskTellFewShotTopk._bert_search.  Everything else (sample, choose, reset,
iteration) is identical to the original Pool.

The original pool.py is preserved and unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, List

import numpy as np


class BertPool:
    """Pool of unlabelled candidates with no embedding dependencies.

    approx_sample is intentionally absent — similarity search is handled
    upstream by BertAskTellFewShotTopk._bert_search using materialBERT
    embeddings and a pre-computed cache.

    Example
    -------
    >>> pool = BertPool([row_to_x(raw_data.iloc[i]) for i in indexes],
    ...                 formatter=kwargs['x_formatter'])
    >>> pool.sample(3)
    >>> pool.choose(x)
    >>> pool.reset()
    """

    def __init__(
        self,
        pool: List[Any],
        formatter: Callable[[Any], str] = lambda x: str(x),
    ) -> None:
        if not isinstance(pool, list):
            raise TypeError("Pool must be a list")
        self._pool      = pool
        self._selected: List[Any] = []
        self._available = pool[:]
        self.format     = formatter

    def sample(self, n: int) -> List[Any]:
        """Return n items chosen uniformly at random from available items."""
        if n > len(self._available):
            raise ValueError(
                f"Requested {n} items but only {len(self._available)} available."
            )
        return list(np.random.choice(self._available, n, replace=False))

    def choose(self, x: Any) -> None:
        """Mark x as selected and remove it from the available pool."""
        if x not in self._available:
            raise ValueError("Item not in available pool.")
        self._selected.append(x)
        self._available.remove(x)

    def reset(self) -> None:
        """Restore all items to available (e.g. between optimisation runs)."""
        self._selected  = []
        self._available = self._pool[:]

    def __len__(self)  -> int:           return len(self._pool)
    def __iter__(self):                  return iter(self._available)
    def __repr__(self) -> str:           return f"BertPool({len(self)} items, {len(self._selected)} selected)"
    __str__ = __repr__
