# """utilities for building and selecting from a pool"""
# from typing import List, Any, Callable
# import numpy as np
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings


# class Pool:
#     """Class for sampling from pool of possible data points

#     Example:
#         >>> pool = Pool(['a', 'b', 'c', 'd', 'e'])
#         >>> pool.sample(3)
#         ['a', 'd', 'c']
#         >>> pool.choose('a')
#         >>> pool.sample(3)
#         ['b', 'c', 'd']
#         >>> pool.approx_sample('a', 3)
#         ['b', 'c', 'd']
#     """

#     def __init__(self, pool: List[Any], formatter: Callable = lambda x: str(x)) -> None:
#         if type(pool) is not list:
#             raise TypeError("Pool must be a list")
#         self._pool = pool
#         self._selected = []
#         self._available = pool[:]
#         self.format = formatter
#         self._db = FAISS.from_texts(
#             [formatter(x) for x in pool],
#             OpenAIEmbeddings(model="text-embedding-3-large"),  # model="text-embedding-3-large"
#             metadatas=[dict(data=p) for p in pool],
#         )

#     def sample(self, n: int) -> List[str]:
#         """Sample n items from the pool"""
#         if n > len(self._available):
#             raise ValueError("Not enough items in pool")
#         samples = np.random.choice(self._available, size=n, replace=False)
#         return samples

#     def choose(self, x: str) -> None:
#         """Choose a specific item from the pool"""
#         if x not in self._available:
#             raise ValueError("Item not in pool")
#         self._selected.append(x)
#         self._available.remove(x)

#     def approx_sample(self, x: str, k: int, lambda_mult: float = 0.5) -> None:
#         """Given an approximation of x, return k similar"""

#         # want to select extra, then remove previously chosen
#         _k = k + len(self._selected)
#         docs = self._db.max_marginal_relevance_search(
#             x, k=_k, fetch_k=5 * _k, lambda_mult=lambda_mult
#         )
#         docs = [d.metadata["data"] for d in docs]
#         # remove previously chosen
#         docs = [d for d in docs if d not in self._selected]
#         # select k
#         return docs[:k]

#     def reset(self) -> None:
#         """Reset the pool"""
#         self._selected = []
#         self._available = self._pool[:]

#     def __len__(self) -> int:
#         return len(self._pool)

#     def __repr__(self) -> str:
#         return f"Pool of {len(self)} items with {len(self._selected)} selected"

#     def __str__(self) -> str:
#         return f"Pool of {len(self)} items with {len(self._selected)} selected"

#     def __iter__(self):
#         return iter(self._available)

#------


# """
# utilities for building and selecting from a pool
# ------------------------------------------------
# • No LangChain dependency
# • Batched OpenAI embeddings (v1 client)
# • Persistent on-disk cache (cloudpickle)
# • FAISS index for fast cosine/IP retrieval
# """

# from typing import List, Any, Callable, Dict
# import os, time, pickle, numpy as np, faiss, openai
# from tqdm.auto import tqdm     # progress bar; omit if unwanted
# from langchain_community.vectorstores import FAISS


# # ---------------------------------------------------------------------
# # configuration
# # ---------------------------------------------------------------------
# CACHE_DIR   = ".emb_cache"      # folder for embedding caches
# os.makedirs(CACHE_DIR, exist_ok=True)
# CLIENT      = openai.OpenAI()   # v1 client
# BATCH_SIZE  = 100               # safe under 3k RPM soft-limit
# BACKOFF_S   = 30                # seconds to wait on 429
# MODEL       = "text-embedding-3-large"

# # ---------------------------------------------------------------------
# class Pool:
#     """
#     Fast embedding pool using OpenAI + FAISS.
    
#     Example
#     -------
#     >>> pool = Pool(['a', 'b', 'c'])
#     >>> pool.sample(2)
#     ['c', 'a']
#     >>> pool.choose('c')
#     >>> pool.approx_sample('a', 2)
#     ['b']
#     """

#     # --------------- constructor -------------------------------------
#     def __init__(
#         self,
#         pool: List[Any],
#         formatter: Callable[[Any], str] = lambda x: str(x),
#         model: str = MODEL,
#         batch_size: int = BATCH_SIZE,
#         cache_dir: str = CACHE_DIR,
#     ):
#         if not isinstance(pool, list):
#             raise TypeError("Pool must be a list")

#         self._pool:        List[Any]   = pool
#         self._selected:    List[Any]   = []
#         self._available:   List[Any]   = pool[:]
#         self.format:       Callable    = formatter
#         self.model:        str         = model
#         self.batch_size:   int         = batch_size

#         # ---------- 1. load / build embedding cache ----------
#         self._cache_path = os.path.join(
#             cache_dir, f"{model.replace('/', '_')}.pkl"
#         )
#         self._embeds: Dict[str, List[float]] = self._load_cache()

#         texts = [formatter(x) for x in pool]
#         missing = [t for t in texts if t not in self._embeds]

#         if missing:
#             print(f"🚀 Embedding {len(missing):,} new texts …")
#             self._embed_and_cache(missing)

#         # ---------- 2. build FAISS index ----------
#         emb_matrix = np.asarray([self._embeds[self.format(x)] for x in pool],
#                                 dtype="float32")
#         faiss.normalize_L2(emb_matrix)
#         dim = emb_matrix.shape[1]
#         self.index = faiss.IndexFlatIP(dim)
#         self.index.add(emb_matrix)
#         self.text_to_idx = {self.format(x): i for i, x in enumerate(pool)}

#     # --------------- private helpers ---------------------------------
#     def _load_cache(self) -> Dict[str, List[float]]:
#         if os.path.exists(self._cache_path):
#             with open(self._cache_path, "rb") as f:
#                 cache = pickle.load(f)
#             print(f"✅ Loaded cache with {len(cache):,} embeddings.")
#             return cache
#         print("⚠️  No cache found; starting fresh.")
#         return {}

#     def _embed_and_cache(self, texts: List[str]) -> None:
#         for i in tqdm(range(0, len(texts), self.batch_size)):
#             batch = texts[i : i + self.batch_size]
#             done = False
#             while not done:
#                 try:
#                     rsp = CLIENT.embeddings.create(
#                         model=self.model,
#                         input=batch,
#                         encoding_format="float",
#                     )
#                     for t, d in zip(batch, rsp.data):
#                         self._embeds[t] = d.embedding
#                     done = True
#                 except openai.RateLimitError:
#                     print(f"⏳ 429 rate-limited; sleeping {BACKOFF_S}s")
#                     time.sleep(BACKOFF_S)
#         # persist cache
#         with open(self._cache_path, "wb") as f:
#             pickle.dump(self._embeds, f)
#         print(f"💾 Cache saved ({len(self._embeds):,} total embeddings).")

#     # --------------- public API --------------------------------------
#     def sample(self, n: int) -> List[Any]:
#         if n > len(self._available):
#             raise ValueError("Not enough items in pool")
#         return list(np.random.choice(self._available, size=n, replace=False))

#     def choose(self, x: Any) -> None:
#         if x not in self._available:
#             raise ValueError("Item not in pool")
#         self._selected.append(x)
#         self._available.remove(x)

#     # def approx_sample(self, x: Any, k: int) -> List[Any]:
#     #     """Return k pool items most similar to x (excluding already selected)."""
#     #     query_text = self.format(x)
#     #     if query_text not in self.text_to_idx:
#     #         raise ValueError("Reference item not in pool")

#     #     qvec = np.asarray(
#     #         self._embeds[query_text], dtype="float32"
#     #     ).reshape(1, -1)
#     #     faiss.normalize_L2(qvec)
#     #     _, I = self.index.search(qvec, k + len(self._selected) + 1)

#     #     results = [
#     #         self._pool[i]
#     #         for i in I[0]
#     #         if self._pool[i] not in self._selected and self._pool[i] != x
#     #     ]
#     #     return results[:k]
#     def approx_sample(self, x: str, k: int, lambda_mult: float = 0.5) -> None:

#         """Given an approximation of x, return k similar"""

#         # want to select extra, then remove previously chosen
#         _k = k + len(self._selected)
#         docs = self._db.max_marginal_relevance_search(
#             x, k=_k, fetch_k=5 * _k, lambda_mult=lambda_mult
#         )
#         docs = [d.metadata["data"] for d in docs]
#         # remove previously chosen
#         docs = [d for d in docs if d not in self._selected]
#         # select k
#         return docs[:k]

#     def reset(self) -> None:
#         self._selected = []
#         self._available = self._pool[:]

#     # --------------- dunder methods ----------------------------------
#     def __len__(self) -> int:
#         return len(self._pool)

#     def __iter__(self):
#         return iter(self._available)

#     def __repr__(self) -> str:
#         return f"Pool of {len(self)} items, {len(self._selected)} selected"

#     __str__ = __repr__
"""
utilities for building and selecting from a pool
------------------------------------------------
• No LangChain dependency
• Batched OpenAI embeddings (v1 client)
• Persistent on-disk cache (pickle)
• FAISS index for fast cosine/IP retrieval
• Greedy MMR with tunable lambda_mult
"""

from typing import List, Any, Callable, Dict
import os, time, pickle, numpy as np, faiss, openai
from tqdm.auto import tqdm         # IProgress warning is harmless

# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------
CACHE_DIR  = ".emb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

CLIENT      = openai.OpenAI()      # openai>=1.0
MODEL       = "text-embedding-3-large"
BATCH_SIZE  = 100                  # safe under 3k RPM soft-limit
BACKOFF_S   = 30                   # seconds to wait on HTTP 429

# ---------------------------------------------------------------------
class Pool:
    """
    Fast embedding pool using OpenAI + FAISS (+ greedy MMR).

    Example
    -------
    >>> pool = Pool(['a', 'b', 'c'])
    >>> pool.sample(2)
    ['c', 'a']
    >>> pool.choose('c')
    >>> pool.approx_sample('a', 1, lambda_mult=0.3)
    ['b']
    """

    # ---------------- constructor -----------------------------------
    def __init__(
        self,
        pool: List[Any],
        formatter: Callable[[Any], str] = lambda x: str(x),
        model: str   = MODEL,
        batch_size: int = BATCH_SIZE,
        cache_dir:  str = CACHE_DIR,
    ):
        if not isinstance(pool, list):
            raise TypeError("Pool must be a list")

        self._pool        = pool
        self._selected    = []
        self._available   = pool[:]
        self.format       = formatter
        self.model        = model
        self.batch_size   = batch_size

        # -------- 1. load / build embedding cache -------------------
        cache_path = os.path.join(cache_dir, f"{model.replace('/','_')}.pkl")
        self._embeds: Dict[str, List[float]] = (
            pickle.load(open(cache_path, "rb"))
            if os.path.exists(cache_path) else {}
        )
        print(
            f"{'✅' if self._embeds else '⚠️'} "
            f"{'Loaded' if self._embeds else 'No'} cache "
            f"({len(self._embeds):,} embeddings)."
        )

        texts   = [formatter(x) for x in pool]
        missing = [t for t in texts if t not in self._embeds]
        if missing:
            print(f"🚀 Embedding {len(missing):,} new texts …")
            self._embed_and_cache(missing, cache_path)

        # -------- 2. build FAISS index ------------------------------
        emb_matrix = np.asarray(
            [self._embeds[formatter(x)] for x in pool], dtype="float32"
        )
        faiss.normalize_L2(emb_matrix)
        dim = emb_matrix.shape[1]

        self.index        = faiss.IndexFlatIP(dim)
        self.index.add(emb_matrix)
        self._emb_matrix  = emb_matrix          # keep for queries
        self.text_to_idx  = {formatter(x): i for i, x in enumerate(pool)}

        # free the heavy dict payload to save RAM
        for k in self._embeds.keys():
            self._embeds[k] = None

    # ---------------- batching helper -------------------------------
    def _embed_and_cache(self, texts, cache_path):
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i : i + self.batch_size]
            while True:
                try:
                    rsp = CLIENT.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float",
                    )
                    break
                except openai.RateLimitError:
                    print(f"⏳ 429 – sleeping {BACKOFF_S}s")
                    time.sleep(BACKOFF_S)
            for t, d in zip(batch, rsp.data):
                self._embeds[t] = d.embedding

        with open(cache_path, "wb") as f:
            pickle.dump(self._embeds, f)
        print(f"💾 Cache saved ({len(self._embeds):,} total).")

    # ---------------- MMR helper ------------------------------------
    def _mmr(self, qvec, cand_idxs, k, lambda_mult=0.5):
        """Greedy Max-Marginal-Relevance selection (cosine/IP)."""
        selected, selected_vecs = [], []
        cand_vecs = self._emb_matrix[cand_idxs]
        sim_q     = cand_vecs @ qvec.T                    # (N,1)

        first = int(np.argmax(sim_q))
        selected.append(cand_idxs[first])
        selected_vecs.append(cand_vecs[first])

        while len(selected) < k and len(selected) < len(cand_idxs):
            remaining = [idx for idx in cand_idxs if idx not in selected]
            rem_vecs  = self._emb_matrix[remaining]

            sim_query = rem_vecs @ qvec.T
            sim_div   = rem_vecs @ np.vstack(selected_vecs).T
            max_div   = sim_div.max(axis=1, keepdims=True)

            mmr = lambda_mult * sim_query - (1 - lambda_mult) * max_div
            next_idx = remaining[int(np.argmax(mmr))]
            selected.append(next_idx)
            selected_vecs.append(self._emb_matrix[next_idx])

        return selected

    # ---------------- public API ------------------------------------
    def sample(self, n: int) -> List[Any]:
        if n > len(self._available):
            raise ValueError("Not enough items in pool")
        return list(np.random.choice(self._available, n, replace=False))

    def choose(self, x: Any) -> None:
        if x not in self._available:
            raise ValueError("Item not in pool")
        self._selected.append(x)
        self._available.remove(x)

    # ---------------- alternate constructor -------------------------
    @classmethod
    def from_prebuilt(
        cls,
        prompts_path: str,
        emb_path: str,
        index_path: str,
        formatter: Callable[[Any], str] = lambda x: str(x),
        model: str = MODEL,                 #  ← add default
    ):
        import pickle, faiss, numpy as np

        with open(prompts_path, "rb") as f:
            prompt_list = pickle.load(f)
        emb_matrix = np.load(emb_path, mmap_mode="r")
        index      = faiss.read_index(index_path)

        self = cls.__new__(cls)            # bypass __init__

        # fields needed by all methods
        self._pool        = prompt_list
        self._selected    = []
        self._available   = prompt_list[:]
        self.format       = formatter
        self._emb_matrix  = emb_matrix
        self.index        = index
        self.text_to_idx  = {formatter(x): i for i, x in enumerate(prompt_list)}

        # ------ add these two lines ------
        self.model        = model          # so approx_sample can embed a query
        self.batch_size   = BATCH_SIZE     # not critical but keeps parity
        # ---------------------------------

        return self


    # def approx_sample(
    #     self,
    #     x: Any,
    #     k: int,
    #     fetch_k: int = 100,
    #     lambda_mult: float = 0.5,
    # ) -> List[Any]:
    #     """
    #     Return *k* items similar to *x*, re-ranked via MMR.
    #     lambda_mult → 1.0 = pure relevance; 0.0 = max diversity.
    #     """
    #     qtxt = self.format(x)
    #     if qtxt not in self.text_to_idx:
    #         raise ValueError("Reference item not in pool")

    #     qvec = self._emb_matrix[self.text_to_idx[qtxt]].reshape(1, -1)
    #     _, idxs = self.index.search(qvec, fetch_k + len(self._selected) + 1)
    #     cand = [
    #         i for i in idxs[0]
    #         if self._pool[i] not in self._selected and self._pool[i] != x
    #     ]
    #     chosen = self._mmr(qvec, cand, k, lambda_mult=lambda_mult)
    #     return [self._pool[i] for i in chosen]
    def approx_sample(
        self,
        x: Any,
        k: int,
        fetch_k: int = 100,
        lambda_mult: float = 0.5,
    ) -> List[Any]:
        """
        Return *k* items similar to *x* (string or pool item), MMR-reranked.
        If *x* is not in the pool we embed it on the fly.
        """
        qtxt = self.format(x)

        # ---------- embed query if it's new ----------
        if qtxt in self.text_to_idx:
            qvec = self._emb_matrix[self.text_to_idx[qtxt]].reshape(1, -1)
        else:
            rsp  = CLIENT.embeddings.create(
                model=self.model,
                input=[qtxt],
                encoding_format="float",
            )
            qvec = np.asarray(rsp.data[0].embedding, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(qvec)

        # ---------- FAISS search + MMR ---------------
        _, idxs = self.index.search(qvec, fetch_k + len(self._selected) + 1)
        cand = [
            i for i in idxs[0]
            if self._pool[i] not in self._selected and self._pool[i] != x
        ]
        chosen = self._mmr(qvec, cand, k, lambda_mult=lambda_mult)
        return [self._pool[i] for i in chosen]


    def reset(self):          self._selected, self._available = [], self._pool[:]
    def __len__(self):        return len(self._pool)
    def __iter__(self):       return iter(self._available)
    def __repr__(self):       return f"Pool of {len(self)} items, {len(self._selected)} selected"
    __str__ = __repr__
