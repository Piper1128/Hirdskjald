"""Dense embeddings (semantisk opslag) med en lille flersproget e5-model.

Bruger transformers direkte — ingen sentence-transformers-dep. Modellen er lille
(~120 MB) og kører på GPU ved siden af skjalden. e5 kræver 'query:'/'passage:'-præfiks.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EMB_MODEL = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name=EMB_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts, prefix="passage: ", batch_size=64, max_length=384):
        """Returnér L2-normaliserede vektorer (N, d) på CPU. Cosine = prik-produkt."""
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean-pool
            vecs.append(F.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(vecs, dim=0) if vecs else torch.empty(0)

    def embed_query(self, text):
        return self.embed([text], prefix="query: ")[0]
