"""RAG-retrieval over den danske viden-base (data/knowledge.jsonl).

To metoder:
  * dense (semantisk) via e5-embeddings — bruges når en Embedder gives (bedst recall)
  * BM25 (nøgleord, ren Python) — fallback hvis ingen embedder

Hent basen med scripts/fetch_knowledge.py.
"""
import json
import math
import os
import re
from collections import Counter

import torch

_WORD = re.compile(r"[0-9a-zæøåA-ZÆØÅ]+")


def _tok(s):
    return [w.lower() for w in _WORD.findall(s)]


# Middelalderlig kilde (bog/skrift) bag hvert videns-emne — så skjalden kan sige hvorfra.
_MEDIEVAL_SOURCES = {
    "Vøluspá": "Vøluspá i den Poetiske Edda",
    "Vølvens spådom": "Vøluspá i den Poetiske Edda",
    "Håvamål": "Håvamål i den Poetiske Edda",
    "Den ældre Edda": "den Poetiske (Ældre) Edda, ca. 1270",
    "Den yngre Edda": "Snorris (Yngre) Edda, ca. 1220",
    "Snorris Edda": "Snorris (Yngre) Edda, ca. 1220",
    "Egils saga": "Egils saga Skallagrímssonar (islandsk slægtssaga, ca. 1230)",
    "Njáls saga": "Njáls saga (islandsk slægtssaga, ca. 1280)",
    "Vølsunga saga": "Vølsunga saga (islandsk fornaldersaga)",
    "Gunlaug Ormstunges saga": "Gunnlaugs saga ormstungu",
    "Regnar Lodbrog": "Ragnars saga loðbrókar og Saxos Gesta Danorum",
    "Ragnar Lodbrog": "Ragnars saga loðbrókar og Saxos Gesta Danorum",
    "Runealfabet": "runeindskrifter og middelalderlige kilder",
    "Runer": "runeindskrifter og middelalderlige kilder",
    "Blot": "middelalderlige krøniker og arkæologiske fund",
    "Vikingetiden": "sagaer, krøniker og arkæologiske fund",
    "Holmgang": "de islandske sagaer",
    "Ting (forsamling)": "de islandske sagaer og gamle lovtekster",
    "Skjald": "skjaldekvadene og sagaerne",
    "Bersærk": "sagaerne og Snorris Edda",
}


def medieval_source(title):
    """Den middelalderlige kilde (bog/skrift) bag et videns-emne."""
    if title in _MEDIEVAL_SOURCES:
        return _MEDIEVAL_SOURCES[title]
    if "saga" in title.lower():
        return title + " (islandsk saga)"
    return "Snorris Edda og den Poetiske Edda"  # de fleste myter


class Knowledge:
    """Opslag i viden-basen. Tom (falsy) hvis basen ikke er hentet."""

    def __init__(self, path, embedder=None, cache_path=None):
        self.chunks = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.chunks.append(json.loads(line))
        self.N = len(self.chunks)

        # BM25-indeks (altid bygget — billigt og bruges som fallback)
        self.docs = [_tok((c["title"] + " ") * 3 + c["text"]) for c in self.chunks]
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 1.0
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {w: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

        # Dense (semantisk) indeks hvis en embedder gives
        self.embedder = embedder
        self.emb = None
        if embedder is not None and self.N:
            self._build_dense(cache_path or (path + ".emb.pt"))

    def _build_dense(self, cache_path):
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path)
                if data.get("n") == self.N:
                    self.emb = data["emb"]
                    return
            except Exception:
                pass
        texts = [c["title"] + ". " + c["text"] for c in self.chunks]
        self.emb = self.embedder.embed(texts)  # (N, d) L2-normaliseret, CPU
        try:
            torch.save({"n": self.N, "emb": self.emb}, cache_path)
        except Exception:
            pass

    def __bool__(self):
        return self.N > 0

    def retrieve(self, query, k=4, min_score=0.79):
        """Top-k (score, chunk). Semantisk hvis muligt, ellers BM25."""
        if not self.N:
            return []
        if self.emb is not None:
            q = self.embedder.embed_query(query)          # (d,)
            sims = self.emb @ q                            # cosine (begge normaliseret)
            vals, idx = torch.topk(sims, min(k, self.N))
            return [(float(v), self.chunks[int(i)])
                    for v, i in zip(vals.tolist(), idx.tolist()) if v >= min_score]
        return self._bm25(query, k)

    def _bm25(self, query, k, k1=1.5, b=0.75, min_score=3.5):
        q = _tok(query)
        scored = []
        for i in range(self.N):
            tf = self.tf[i]
            dl = len(self.docs[i])
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += (self.idf.get(w, 0.0) * f * (k1 + 1)
                          / (f + k1 * (1 - b + b * dl / self.avgdl)))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scored[:k] if s >= min_score]
