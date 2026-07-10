"""RAG-retrieval: BM25 over den danske viden-base (data/knowledge.jsonl).

Ren Python — ingen embedding-model at downloade. Nøgleord-baseret opslag er
stærkt for et myte-korpus, hvor spørgsmål deler ord med kilden (Odin, Ragnarok,
Yggdrasil ...). Hent basen med scripts/fetch_knowledge.py.
"""
import json
import math
import os
import re
from collections import Counter

_WORD = re.compile(r"[0-9a-zæøåA-ZÆØÅ]+")


def _tok(s):
    return [w.lower() for w in _WORD.findall(s)]


class Knowledge:
    """BM25-indeks over viden-stykkerne. Tom (falsy) hvis basen ikke er hentet."""

    def __init__(self, path):
        self.chunks = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.chunks.append(json.loads(line))
        # Titlen tælles tredobbelt, så et opslag rammer den rette artikel hårdere.
        self.docs = [_tok((c["title"] + " ") * 3 + c["text"])
                     for c in self.chunks]
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 1.0
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {w: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
                    for w, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def __bool__(self):
        return self.N > 0

    def retrieve(self, query, k=3, k1=1.5, b=0.75, min_score=3.5):
        """Returnér op til k (score, chunk) over min_score, bedste først."""
        if not self.N:
            return []
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
