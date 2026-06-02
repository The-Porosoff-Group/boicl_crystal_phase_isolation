"""
bert_ask_tell.py
================
Extends AskTellFewShotTopk so that the pool-search step in ``ask`` uses
materialBERT embeddings instead of the Ada-based ``Pool.approx_sample``.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .asktell import AskTellFewShotTopk
from .bert_pool import BertPool
from .aqfxns import (
    expected_improvement,
    greedy,
    log_expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)


class BertAskTellFewShotTopk(AskTellFewShotTopk):

    def __init__(
        self,
        *args,
        embedding_cache: Union[Dict[str, np.ndarray], str, os.PathLike],
        bert_model_name: str = "alan-yahya/MatBERT",
        bert_device: Optional[str] = None,
        bert_batch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._bert_model_name = bert_model_name
        self._bert_batch_size = bert_batch_size
        self._bert_device = bert_device
        self._bert_tokenizer = None
        self._bert_model = None

        if isinstance(embedding_cache, (str, os.PathLike)):
            data = np.load(embedding_cache, allow_pickle=True)
            self._embedding_cache: Dict[str, np.ndarray] = dict(
                zip(data["procedures"].tolist(), data["embeddings"])
            )
        else:
            self._embedding_cache = embedding_cache

        self._norm_cache: Dict[str, np.ndarray] = {
            proc: self._l2_norm(emb)
            for proc, emb in self._embedding_cache.items()
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _to_procedure_str(self, x: Any) -> str:
        if isinstance(x, dict):
            return x.get("procedure", self.format_x(x))
        return x

    # ------------------------------------------------------------------
    # materialBERT
    # ------------------------------------------------------------------

    def _load_bert(self) -> None:
        if self._bert_model is not None:
            return
        import torch
        from transformers import BertTokenizer, BertConfig, BertModel

        if self._bert_device is None:
            self._bert_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._bert_tokenizer = BertTokenizer.from_pretrained(self._bert_model_name)

        model_bin = os.path.join(self._bert_model_name, "pytorch_model.bin")
        if os.path.exists(model_bin):
            # local path — bypass torch version check
            config = BertConfig.from_pretrained(self._bert_model_name)
            self._bert_model = BertModel(config)
            state_dict = torch.load(model_bin, map_location="cpu", weights_only=False)
            self._bert_model.load_state_dict(state_dict, strict=False)
        else:
            # HuggingFace hub
            self._bert_model = BertModel.from_pretrained(
                self._bert_model_name, add_pooling_layer=False
            )

        self._bert_model.eval()
        self._bert_model.to(self._bert_device)

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a single string and return an L2-normalised vector."""
        import torch

        self._load_bert()
        enc = self._bert_tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._bert_device)

        with torch.no_grad():
            out = self._bert_model(**enc)

        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self._l2_norm(emb.squeeze(0).cpu().numpy())

    # ------------------------------------------------------------------
    # Pool search
    # ------------------------------------------------------------------

    def _bert_search(self, query_text: str, pool: BertPool, k: int) -> List[Any]:
        pool_items = list(pool)
        if not pool_items:
            return []

        pool_vecs: List[np.ndarray] = []
        valid_items: List[Any] = []
        missing: List[str] = []

        for item in pool_items:
            proc = self._to_procedure_str(item)
            if proc in self._norm_cache:
                pool_vecs.append(self._norm_cache[proc])
                valid_items.append(item)
            else:
                missing.append(proc[:60])

        if missing:
            import warnings
            warnings.warn(
                f"BertAskTellFewShotTopk: {len(missing)} pool item(s) not found "
                f"in embedding cache and will be skipped.\n"
                f"First missing: {missing[0]!r}",
                RuntimeWarning,
                stacklevel=2,
            )

        if not valid_items:
            return []

        query_vec = self._embed_query(query_text)
        pool_matrix = np.vstack(pool_vecs)
        sims = pool_matrix @ query_vec
        top_k_idx = np.argsort(sims)[::-1][: min(k, len(valid_items))]
        return [valid_items[i] for i in top_k_idx]

    # ------------------------------------------------------------------
    # ask override
    # ------------------------------------------------------------------

    def ask(
        self,
        possible_x: Union[BertPool, List[Any]],
        aq_fxn: str = "upper_confidence_bound",
        k: int = 1,
        inv_filter: int = 16,
        aug_random_filter: int = 0,
        lambda_mult: float = 0.5,
        _lambda: float = 0.5,
        system_message: Optional[str] = "",
        inv_system_message: Optional[str] = "",
    ) -> Tuple[List[Any], List[float], List[float], List[float]]:
        if isinstance(possible_x, list):
            possible_x = BertPool(possible_x, self.format_x)

        if self._example_count < 2:
            return possible_x.sample(k), [0] * k, [0] * k, [0] * k

        _AQ_MAP = {
            "probability_of_improvement": probability_of_improvement,
            "expected_improvement":       expected_improvement,
            "log_expected_improvement":   log_expected_improvement,
            "greedy":                     greedy,
        }
        if aq_fxn == "upper_confidence_bound":
            aq_fn: Callable = partial(upper_confidence_bound, _lambda=_lambda)
        elif aq_fxn == "random":
            return possible_x.sample(k), [0] * k, [0] * k, [0] * k
        elif aq_fxn in _AQ_MAP:
            aq_fn = _AQ_MAP[aq_fxn]
        else:
            raise ValueError(f"Unknown acquisition function: {aq_fxn!r}")

        best = float(np.max(self._ys)) if self._ys else 0.0

        if inv_filter + aug_random_filter < len(possible_x):
            subpool: List[Any] = []

            if inv_filter:
                target_value = best + np.abs(best) * np.random.normal(1.2, 0.05)
                print("calling inv_predict...", flush=True)
                approx_procedure: str = self.inv_predict(
                    target_value,
                    system_message=inv_system_message,
                )
                print(f"inv_predict done: {approx_procedure[:60]}", flush=True)
                print("calling _bert_search...", flush=True)
                subpool.extend(
                    self._bert_search(approx_procedure, possible_x, inv_filter)
                )
                print("_bert_search done", flush=True)

            if aug_random_filter:
                subpool.extend(possible_x.sample(aug_random_filter))
        else:
            subpool = list(possible_x)

        results = self._ask(subpool, best, aq_fn, k, system_message=system_message)

        if not results[0]:
            return possible_x.sample(k), [0] * k, [0] * k, [0] * k

        return results