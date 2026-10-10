# linguaparity

LinguaParity measures whether multilingual, retrieval-augmented AI systems give
speakers of different languages equal access to information. Instead of scoring
language skills, it follows one information need through the pipeline and asks,
stage by stage, where speakers of lower-resource languages lose out.

Full paper (ICAAD 2026, Konya): `paper/LinguaParity_ICAAD26_full_paper.docx`

## Design

* 32 parallel information needs in four social domains (education, employment,
  public services, digital safety), each tied to a canonical English Wikipedia
  article and three key facts (`data/information_needs.json`).
* Six languages across resource classes: English, Turkish, Urdu, Swahili,
  Yoruba, Amharic.
* Stages measured:
  1. evidence availability (does the language edition have the article?)
  2. retrieval (model-independent lexical search + LaBSE reranking, hit@3)
  3. evidence coverage of key facts for native, English-pivot and mixed evidence
  4. source-language dependence (English share of a mixed ranking)
  5. token cost of the same question

## Headline results

| Language | Article exists | Native hit@3 | Native coverage | Pivot coverage | EN share (mixed) | Tokens vs EN |
|---|---|---|---|---|---|---|
| English | 1.00 | 0.97 | 0.95 | 0.95 | - | 1.00 |
| Turkish | 0.94 | 0.47 | 0.54 | 0.78 | 0.18 | 1.51 |
| Urdu | 0.59 | 0.38 | 0.50 | 0.83 | 0.16 | 3.09 |
| Swahili | 0.25 | 0.09 | 0.21 | 0.69 | 0.45 | 2.04 |
| Yoruba | 0.00 | 0.00 | 0.05 | 0.38 | 0.60 | 3.21 |
| Amharic | 0.03 | 0.03 | 0.05 | 0.66 | 0.54 | 3.99 |

The generation stage (answer completeness, consistency, citations,
hallucination) is part of the framework but not reported yet.

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install transformers sentencepiece sentence-transformers python-docx scikit-learn scipy pandas matplotlib requests
export PYTHONPATH=.
python scripts/01_collect_evidence.py
python scripts/02_translate.py
python scripts/03_retrieve.py
python scripts/04_evidence_eval.py
python scripts/05_analyze.py
python paper/build_paper.py path/to/ICAAD26_Full_Paper_Template.docx
```

Wikipedia responses are cached under `data/cache/` on first run.
