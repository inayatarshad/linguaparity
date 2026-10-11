"""Stage 6: regenerate the result tables in README.md from results/.

Each table lives between <!-- name:start --> and <!-- name:end --> markers,
so the README always matches the data.
"""
import re

import pandas as pd

from linguaparity.config import LANG_CODES, LANGUAGES, RESULTS, ROOT
from linguaparity.io import read_json

L = pd.read_csv(RESULTS / "summary_language.csv").set_index("lang")
S = read_json(RESULTS / "stats.json")
wstats = read_json(RESULTS / "wiki_stats.json")
SCRIPTS = {"en": "Latin", "es": "Latin", "nl": "Latin", "tr": "Latin", "ur": "Perso-Arabic", "sw": "Latin",
           "yo": "Latin", "am": "Ethiopic"}


def f2(v):
    return f"{v:.2f}"


def p_fmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


tables = {}
rows = ["| Language | Script | Resource class | Wikipedia articles (at time of run) |", "|---|---|---|---|"]
for l in LANG_CODES:
    rows.append(f"| {LANGUAGES[l]['name']} | {SCRIPTS.get(l, '')} | {LANGUAGES[l]['joshi']} | "
                f"{wstats[l]['articles']:,} |")
tables["languages"] = rows

rows = ["| Language | Article exists | Native hit@3 | Native coverage | Pivot coverage | Mixed coverage | "
        "EN share in mixed | Tokens vs EN |", "|---|---|---|---|---|---|---|---|"]
for l in LANG_CODES:
    r = L.loc[l]
    rows.append(f"| {r['name']} | {f2(r['article_exists'])} | {f2(r['native_hit3'])} | {f2(r['native_cov'])} | "
                f"{f2(r['pivot_cov'])} | {f2(r['mixed_cov'])} | {'-' if l == 'en' else f2(r['mixed_en_share'])} | "
                f"{f2(r['tok_qwen25'])} |")
tables["main"] = rows

rows = ["| Language | Native | English pivot | Gain | Holm p |", "|---|---|---|---|---|"]
for r in S["pivot_vs_native"]:
    rows.append(f"| {LANGUAGES[r['lang']]['name']} | {f2(r['native'])} | {f2(r['pivot'])} | "
                f"{r['diff']:+.2f} | {p_fmt(r['p_holm'])} |")
tables["pivot"] = rows

dom = S["domain_native_coverage"]
domains = list(next(iter(dom.values())))
rows = ["| Domain | " + " | ".join(l.upper() for l in LANG_CODES) + " |", "|---" * (len(LANG_CODES) + 1) + "|"]
for d in domains:
    rows.append(f"| {d.replace('_', ' ').capitalize()} | " + " | ".join(f2(dom[l][d]) for l in LANG_CODES) + " |")
tables["domains"] = rows

path = ROOT / "README.md"
text = path.read_text(encoding="utf8")
for name, rows in tables.items():
    pattern = re.compile(rf"(<!-- {name}:start -->\n).*?(\n<!-- {name}:end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README is missing the {name} markers")
    text = pattern.sub(lambda m: m.group(1) + "\n".join(rows) + m.group(2), text)
path.write_text(text, encoding="utf8")
print("README tables updated:", ", ".join(tables))
