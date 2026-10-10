"""Thin wrappers around the evaluation models (all run locally on CPU)."""
from functools import lru_cache

import torch

from .config import EMBED_MODEL, LANGUAGES, MT_MODEL

torch.set_num_threads(6)


@lru_cache(None)
def _mt():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MT_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL).eval()
    return tok, model


@torch.inference_mode()
def translate(texts, src, tgt, batch_size=24, max_len=200):
    """NLLB-200 translation between two LinguaParity language codes."""
    if src == tgt:
        return list(texts)
    tok, model = _mt()
    tok.src_lang = LANGUAGES[src]["nllb"]
    tgt_id = tok.convert_tokens_to_ids(LANGUAGES[tgt]["nllb"])
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=1, max_length=None,
                             max_new_tokens=128, no_repeat_ngram_size=4)
        out.extend(zip(idx, tok.batch_decode(gen, skip_special_tokens=True)))
    return [t for _, t in sorted(out)]


@lru_cache(None)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts):
    return _embedder().encode(list(texts), batch_size=32, normalize_embeddings=True,
                              convert_to_numpy=True)
