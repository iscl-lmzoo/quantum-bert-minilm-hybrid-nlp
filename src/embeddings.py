"""
Embeddings utilities with graceful fallbacks.
Priority: SentenceTransformers -> HF Transformers -> simple bag-of-words fallback.
"""
from typing import List
import numpy as np

_st_model = None
_hf_model = None
_hf_tokenizer = None

def _try_sentence_transformers(model_name: str):
    global _st_model
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(model_name)
        return True
    except Exception:
        return False

def _try_hf_transformers(model_name: str):
    global _hf_model, _hf_tokenizer
    try:
        from transformers import AutoModel, AutoTokenizer
        _hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _hf_model = AutoModel.from_pretrained(model_name)
        return True
    except Exception:
        return False

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def embed_texts(texts: List[str],
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                hf_fallback: str = "distilbert-base-uncased") -> np.ndarray:
    """Return (N, D) embeddings for N input texts with graceful fallbacks."""
    assert isinstance(texts, list) and all(isinstance(t, str) for t in texts), "texts must be List[str]"

    # Try SentenceTransformers
    if _try_sentence_transformers(model_name):
        try:
            arr = _st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            pass

    # HF Transformers fallback
    if _try_hf_transformers(hf_fallback):
        try:
            import torch
            _hf_model.eval()
            with torch.no_grad():
                toks = _hf_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                out = _hf_model(**toks)
                pooled = _mean_pool(out.last_hidden_state, toks["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                return pooled.cpu().numpy().astype("float32")
        except Exception:
            pass

    # Simple fallback: hashed bag-of-words
    def bow_vec(s: str, dim: int = 256):
        vec = np.zeros(dim, dtype=np.float32)
        for tok in s.lower().split():
            vec[hash(tok) % dim] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec

    return np.stack([bow_vec(t) for t in texts], axis=0)
