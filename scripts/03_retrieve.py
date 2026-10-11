"""Stage 3: retrieval and source-language distribution.

The retriever is model independent so that it measures what each language
edition of Wikipedia can supply, not what a particular LLM does with it:

1. lexical candidates: the question's tokens are OR-ed into a CirrusSearch
   query against the Wikipedia edition (top CANDIDATES pages);
2. dense reranking: page intros are reranked by LaBSE similarity to the
   question and the TOP_K best become evidence passages.

Three evidence conditions are built per (need, language):
* native : evidence from the language's own Wikipedia;
* pivot  : the question is machine-translated to English and evidence comes
           from English Wikipedia (dependence on a high-resource source);
* mixed  : native and pivot candidates pooled and reranked together; the
           share of English passages that win is the source-language
           distribution.
"""
import re

import numpy as np

from linguaparity import wiki
from linguaparity.config import CANDIDATES, LANG_CODES, PASSAGE_CHARS, RESULTS, TOP_K, load_needs
from linguaparity.io import read_json, read_jsonl, write_jsonl
from linguaparity.models import embed


def or_query(q):
    toks = [t for t in re.findall(r"\w+", q.lower()) if len(t) >= 3]
    return " OR ".join(dict.fromkeys(toks))


def candidates(lang, query):
    titles = wiki.search(lang, or_query(query), CANDIDATES)
    texts = {}
    for b in range(0, len(titles), 20):
        texts.update(wiki.intros(lang, titles[b:b + 20]))
    return [{"lang": lang, "title": t, "text": texts[t][:PASSAGE_CHARS]}
            for t in titles if texts.get(t, "").strip()]


def rerank(query, cands, k):
    if not cands:
        return []
    sims = embed([c["text"] for c in cands]) @ embed([query])[0]
    order = np.argsort(-sims)[:k]
    return [{**cands[i], "score": float(sims[i])} for i in order]


needs = load_needs()
questions = read_json(RESULTS / "questions.json")
avail = {(a["id"], a["lang"]): a for a in read_json(RESULTS / "availability.json")}
# Incremental: (need, language) pairs that already have results are reused unchanged.
done = {}
if (RESULTS / "retrieval.jsonl").exists():
    done = {(r["id"], r["lang"]): r for r in read_jsonl(RESULTS / "retrieval.jsonl")}
rows = []
for n in needs:
    for l in LANG_CODES:
        if (n["id"], l) in done:
            rows.append(done[(n["id"], l)])
            continue
        q, bt = questions[n["id"]][l]["q"], questions[n["id"]][l]["bt"]
        nat_c = candidates(l, q)
        piv_c = nat_c if l == "en" else candidates("en", bt)
        native = rerank(q, nat_c, TOP_K)
        pivot = rerank(q, piv_c, TOP_K)
        mixed = rerank(q, native + ([] if l == "en" else pivot), TOP_K)
        canon_l = avail[(n["id"], l)]["title"]
        rows.append({
            "id": n["id"], "domain": n["domain"], "lang": l,
            "native": native, "pivot": pivot, "mixed": mixed,
            "native_candidates": len(nat_c),
            "native_hit": canon_l is not None and any(p["title"] == canon_l for p in native),
            "pivot_hit": any(p["title"] == n["title"] for p in pivot),
            "mixed_en_share": float(np.mean([p["lang"] == "en" for p in mixed])) if mixed else 0.0,
        })
    print(n["id"], flush=True)
write_jsonl(RESULTS / "retrieval.jsonl", rows)
for l in LANG_CODES:
    r = [x for x in rows if x["lang"] == l]
    print(l, "hit@3 native", np.mean([x["native_hit"] for x in r]).round(2),
          "pivot", np.mean([x["pivot_hit"] for x in r]).round(2),
          "EN share in mixed", np.mean([x["mixed_en_share"] for x in r]).round(2))
