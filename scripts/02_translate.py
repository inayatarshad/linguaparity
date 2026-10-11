"""Stage 2: build the parallel question set and the measurement ceiling.

* Questions are machine-translated from English into each language with
  NLLB-200 and back-translated for quality control.
* The first sentences of each gold reference are round-tripped
  (EN -> L -> EN). Scoring these round-trips with the same metric used for
  model answers gives a per-language ceiling: the score a perfect answer
  would receive once it has passed through the back-translation step.
"""
import numpy as np

from linguaparity.config import LANG_CODES, RESULTS, load_needs
from linguaparity.io import read_json, write_json
from linguaparity.models import embed, translate
from linguaparity.text import sentences

needs = load_needs()
refs = read_json(RESULTS / "references_en.json")
q_en = [n["question"] for n in needs]
e_en = embed(q_en)

questions = {n["id"]: {"en": {"q": n["question"], "bt": n["question"]}} for n in needs}
roundtrip = {"en": {n["id"]: " ".join(sentences(refs[n["id"]])[:2]) for n in needs if int(n["id"][2:]) <= 4}}
qc = {}
# Incremental: languages that already have results are reused unchanged.
if (RESULTS / "translation_qc.json").exists():
    qc = read_json(RESULTS / "translation_qc.json")
    questions = read_json(RESULTS / "questions.json")
    roundtrip = read_json(RESULTS / "reference_roundtrip.json")
for l in LANG_CODES[1:]:
    if l in qc:
        continue
    q_l = translate(q_en, "en", l)
    bt = translate(q_l, l, "en")
    for n, a, b in zip(needs, q_l, bt):
        questions[n["id"]][l] = {"q": a, "bt": b}
    qc[l] = {
        "labse_en_vs_l": float(np.mean((e_en * embed(q_l)).sum(1))),
        "labse_en_vs_backtranslation": float(np.mean((e_en * embed(bt)).sum(1))),
    }
    # Gold-reference round trip on the generation subset, batched per language.
    ids = [n["id"] for n in needs if int(n["id"][2:]) <= 4]
    src = [sentences(refs[i])[:2] for i in ids]
    flat = [x for ss in src for x in ss]
    back = iter(translate(translate(flat, "en", l), l, "en"))
    rt = {i: " ".join(next(back) for _ in ss) for i, ss in zip(ids, src)}
    roundtrip[l] = rt
    print(l, qc[l], flush=True)

write_json(RESULTS / "questions.json", questions)
write_json(RESULTS / "translation_qc.json", qc)
write_json(RESULTS / "reference_roundtrip.json", roundtrip)
