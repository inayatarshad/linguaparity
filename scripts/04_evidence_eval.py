"""Stage 4: evidence coverage and tokenization cost.

Evidence coverage asks whether the passages a retrieval-augmented system would
hand to its LLM actually contain the key facts of the information need. Each
English key fact is compared with every sentence of the evidence using LaBSE
cross-lingual similarity, so no translation step sits between the evidence and
the judgement. A fact counts as covered when its best-matching sentence clears
a threshold tau calibrated on English: tau maximises Youden's J between
(fact, own reference article) and (fact, unrelated reference article) pairs.

Tokenization cost is the number of tokens the generator tokenizers spend on
the same question in each language, relative to English.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer

from linguaparity.config import LANG_CODES, RESULTS, SEED, TOKENIZERS, load_needs
from linguaparity.io import read_json, read_jsonl, write_json
from linguaparity.models import embed
from linguaparity.text import sentences

needs = {n["id"]: n for n in load_needs()}
ids = list(needs)
refs = read_json(RESULTS / "references_en.json")
retrieval = read_jsonl(RESULTS / "retrieval.jsonl")
questions = read_json(RESULTS / "questions.json")
rng = np.random.default_rng(SEED)

facts = [(i, f) for i in ids for f in needs[i]["key_facts"]]
fact_emb = dict(zip(facts, embed([f for _, f in facts])))


def best_sim(fact_vecs, sents):
    if not sents:
        return np.zeros(len(fact_vecs))
    return (np.stack(fact_vecs) @ embed(sents).T).max(1)


prev = read_json(RESULTS / "evidence_eval.json") if (RESULTS / "evidence_eval.json").exists() else None

# ---- calibrate tau on English references ----
ref_sents = {i: sentences(refs[i]) for i in ids}
pos, neg = [], []
for i in ids:
    vecs = [fact_emb[(i, f)] for f in needs[i]["key_facts"]]
    other = rng.choice([j for j in ids if j != i])
    pos += list(best_sim(vecs, ref_sents[i]))
    neg += list(best_sim(vecs, ref_sents[other]))
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
s = np.r_[pos, neg]
fpr, tpr, thr = roc_curve(y, s)
k = int(np.argmax(tpr - fpr))
tau = float(thr[k])
calibration = {"tau": tau, "auc": float(roc_auc_score(y, s)), "tpr": float(tpr[k]), "fpr": float(fpr[k]),
               "n_pairs": int(len(y))}
if prev is not None:
    # Incremental runs keep the stored calibration so existing languages are unchanged.
    calibration, tau = prev["calibration"], prev["calibration"]["tau"]
print("calibration", calibration, flush=True)

# ---- coverage of retrieved evidence ----
done = {(c["id"], c["lang"]): c for c in prev["coverage"]} if prev is not None else {}
coverage = []
for r in retrieval:
    if (r["id"], r["lang"]) in done:
        coverage.append(done[(r["id"], r["lang"])])
        continue
    vecs = [fact_emb[(r["id"], f)] for f in needs[r["id"]]["key_facts"]]
    row = {"id": r["id"], "domain": r["domain"], "lang": r["lang"]}
    for cond in ("native", "pivot", "mixed"):
        sents = [x for p in r[cond] for x in sentences(p["text"])]
        row[cond] = float((best_sim(vecs, sents) >= tau).mean())
    coverage.append(row)
    print(r["id"], r["lang"], {c: round(row[c], 2) for c in ("native", "pivot", "mixed")}, flush=True)

# ---- tokenization cost ----
tokens = {}
for name, repo in TOKENIZERS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    count = {l: [len(tok(questions[i][l]["q"]).input_ids) for i in ids] for l in LANG_CODES}
    tokens[name] = {l: float(np.mean(np.array(count[l]) / np.array(count["en"]))) for l in LANG_CODES}
print("token ratio", tokens)

write_json(RESULTS / "evidence_eval.json", {"calibration": calibration, "coverage": coverage,
                                            "token_ratio": tokens})
