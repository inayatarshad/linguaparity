"""Stage 1: evidence availability.

For every information need, fetch the canonical English Wikipedia article
(its intro is the gold reference) and check whether an equivalent article
exists in each other language edition, and how large it is.
"""
from linguaparity import wiki
from linguaparity.config import LANG_CODES, LANGUAGES, RESULTS, load_needs
from linguaparity.io import write_json

needs = load_needs()
stats = {l: {"articles": wiki.article_count(l), **LANGUAGES[l]} for l in LANG_CODES}
references, availability = {}, []
for n in needs:
    en = wiki.page_info("en", n["title"])
    references[n["id"]] = en["extract"]
    for l in LANG_CODES:
        title = n["title"] if l == "en" else en["langlinks"].get(l)
        info = wiki.page_info(l, title) if title else None
        availability.append({
            "id": n["id"], "domain": n["domain"], "lang": l,
            "title": info["title"] if info else None,
            "exists": info is not None,
            "bytes": info["length"] if info else 0,
            "bytes_ratio_to_en": (info["length"] / en["length"]) if info else 0.0,
        })

write_json(RESULTS / "wiki_stats.json", stats)
write_json(RESULTS / "references_en.json", references)
write_json(RESULTS / "availability.json", availability)
for l in LANG_CODES:
    rows = [a for a in availability if a["lang"] == l]
    print(l, stats[l]["articles"], "coverage", sum(a["exists"] for a in rows), "/", len(rows))
