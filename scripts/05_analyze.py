"""Stage 5: aggregate tables, statistics and figures."""
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from linguaparity.config import FIGURES, LANG_CODES, LANGUAGES, RESULTS, SEED  # noqa: E402
from linguaparity.io import read_json, read_jsonl, write_json  # noqa: E402

rng = np.random.default_rng(SEED)


def boot_ci(x, n=5000):
    x = np.asarray(x, dtype=float)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(v) for v in np.percentile(m, [2.5, 97.5])]


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[k]))
        adj[k] = running
    return adj


wstats = read_json(RESULTS / "wiki_stats.json")
qc = read_json(RESULTS / "translation_qc.json")
avail = pd.DataFrame(read_json(RESULTS / "availability.json"))
retr = pd.DataFrame(read_jsonl(RESULTS / "retrieval.jsonl"))
ev = read_json(RESULTS / "evidence_eval.json")
cov = pd.DataFrame(ev["coverage"])

rows = []
for l in LANG_CODES:
    a, r, c = avail[avail.lang == l], retr[retr.lang == l], cov[cov.lang == l]
    rows.append({
        "lang": l, "name": LANGUAGES[l]["name"], "joshi": LANGUAGES[l]["joshi"],
        "wiki_articles": wstats[l]["articles"],
        "article_exists": a.exists.mean(),
        "bytes_ratio": a.bytes_ratio_to_en.mean(),
        "native_hit3": r.native_hit.mean(),
        "native_cov": c["native"].mean(), "native_cov_ci": boot_ci(c["native"]),
        "pivot_cov": c["pivot"].mean(), "pivot_cov_ci": boot_ci(c["pivot"]),
        "mixed_cov": c["mixed"].mean(),
        "mixed_en_share": r.mixed_en_share.mean() if l != "en" else np.nan,
        "mt_labse": qc[l]["labse_en_vs_l"] if l in qc else 1.0,
        "mt_roundtrip_labse": qc[l]["labse_en_vs_backtranslation"] if l in qc else 1.0,
        "tok_qwen25": ev["token_ratio"]["Qwen2.5"][l],
        "tok_qwen3": ev["token_ratio"]["Qwen3"][l],
    })
lang_df = pd.DataFrame(rows)
lang_df.to_csv(RESULTS / "summary_language.csv", index=False)

out = {"calibration": ev["calibration"], "native_vs_english": [], "pivot_vs_native": [], "spearman": []}
en = cov[cov.lang == "en"].set_index("id").native
p1, p2 = [], []
for l in LANG_CODES[1:]:
    c = cov[cov.lang == l].set_index("id")
    d1 = (c["native"] - en.reindex(c.index))
    d2 = (c["pivot"] - c["native"])
    t1 = stats.wilcoxon(d1, zero_method="zsplit").pvalue if (d1 != 0).any() else 1.0
    t2 = stats.wilcoxon(d2, zero_method="zsplit").pvalue if (d2 != 0).any() else 1.0
    p1.append(t1)
    p2.append(t2)
    out["native_vs_english"].append({"lang": l, "diff": float(d1.mean()), "p": float(t1)})
    out["pivot_vs_native"].append({"lang": l, "native": float(c["native"].mean()), "pivot": float(c["pivot"].mean()),
                                   "diff": float(d2.mean()), "p": float(t2)})
for rec, adj in zip(out["native_vs_english"], holm(p1)):
    rec["p_holm"] = float(adj)
for rec, adj in zip(out["pivot_vs_native"], holm(p2)):
    rec["p_holm"] = float(adj)

m = cov.merge(avail[["id", "lang", "exists"]], on=["id", "lang"])
m = m[m.lang != "en"]
rho, p = stats.spearmanr(m.exists.astype(float), m["native"])
out["spearman"].append({"x": "article_exists", "y": "native_coverage", "rho": float(rho), "p": float(p), "n": len(m)})
rho, p = stats.spearmanr(np.log10(lang_df.wiki_articles), lang_df.native_cov)
out["spearman"].append({"x": "log10_wiki_articles", "y": "native_coverage", "rho": float(rho), "p": float(p), "n": len(lang_df)})
rho, p = stats.spearmanr(lang_df.article_exists, lang_df.native_cov)
out["spearman"].append({"x": "article_exists_rate", "y": "native_coverage", "rho": float(rho), "p": float(p), "n": len(lang_df)})
dom = cov.groupby(["domain", "lang"])["native"].mean().unstack()[LANG_CODES]
out["domain_native_coverage"] = dom.round(3).to_dict()
write_json(RESULTS / "stats.json", out)

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
names = [LANGUAGES[l]["name"] for l in LANG_CODES]
x = np.arange(len(LANG_CODES))
fig, ax = plt.subplots(figsize=(6.3, 2.9))
series = [("Canonical article exists", lang_df.article_exists, "#4c72b0", "o", "-"),
          ("Canonical article in native top-3", lang_df.native_hit3, "#55a868", "s", "-"),
          ("Key facts covered, native evidence", lang_df.native_cov, "#c44e52", "^", "-"),
          ("Key facts covered, English-pivot evidence", lang_df.pivot_cov, "#8172b2", "v", "--"),
          ("English share of mixed top-3", lang_df.mixed_en_share, "#333333", "D", ":")]
for label, y, col, mk, ls in series:
    ax.plot(x, y, marker=mk, color=col, ls=ls, label=label, lw=1.2, ms=4)
ax.set_xticks(x, names)
ax.set_ylim(-0.03, 1.05)
ax.set_ylabel("Proportion")
ax.legend(fontsize=6.3, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(FIGURES / "fig1_pipeline.png", dpi=300)

fig, ax = plt.subplots(figsize=(6.3, 1.9))
im = ax.imshow(dom.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(x, names)
ax.set_yticks(range(len(dom.index)), [d.replace("_", " ").capitalize() for d in dom.index])
for a in range(dom.shape[0]):
    for b in range(dom.shape[1]):
        v = dom.values[a, b]
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Native coverage")
fig.tight_layout()
fig.savefig(FIGURES / "fig2_domains.png", dpi=300)

pd.set_option("display.width", 250, "display.max_columns", 30)
print(lang_df.round(3).to_string())
print(pd.DataFrame(out["native_vs_english"]).round(4).to_string())
print(pd.DataFrame(out["pivot_vs_native"]).round(4).to_string())
print(pd.DataFrame(out["spearman"]).round(4).to_string())
print(dom.round(2))
