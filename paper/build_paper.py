"""Build the ICAAD 2026 full paper (.docx) from the template and the results.

Usage: python paper/build_paper.py "D:/ICAAD26_Full_Paper_Template.docx"
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_builder import PaperBuilder  # noqa: E402
from linguaparity.config import FIGURES, LANG_CODES, RESULTS  # noqa: E402
from linguaparity.io import read_json  # noqa: E402

template = sys.argv[1]
out_path = Path(__file__).resolve().parent / "LinguaParity_ICAAD26_full_paper.docx"

L = pd.read_csv(RESULTS / "summary_language.csv").set_index("lang")
S = read_json(RESULTS / "stats.json")
cal = S["calibration"]
nve = {r["lang"]: r for r in S["native_vs_english"]}
pvn = {r["lang"]: r for r in S["pivot_vs_native"]}
sp = {r["x"]: r for r in S["spearman"]}
dom = S["domain_native_coverage"]


def pct(v):
    return f"{100 * v:.0f}%"


def f2(v):
    return f"{v:.2f}"


def p_fmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


exists = {l: int(round(L.loc[l, "article_exists"] * 32)) for l in LANG_CODES}
low = ["sw", "yo", "am"]
low_native = L.loc[low, "native_cov"].mean()
low_pivot = L.loc[low, "pivot_cov"].mean()

b = PaperBuilder(template)
b.title("LinguaParity: Do Multilingual LLMs Provide Equal Access to Information Across Languages?")
b.authors("Inayat Arshad^{1}")
b.affiliation("^{1}Department of Computer and Information Sciences, Pakistan Institute of Engineering and Applied "
              "Sciences (PIEAS), Lehtrar Road, Nilore, Islamabad, Pakistan")
b.affiliation("bscs2248@pieas.edu.pk   |   ORCID: 0009-0006-3999-3144")
b.spacer()

b.abstract(
    "Large language models (LLMs) are marketed as multilingual, yet multilingual capability does not guarantee "
    "that speakers of different languages receive equally reliable and complete information. This study introduces "
    "LinguaParity, a framework that treats multilingual AI as an information-access pipeline and measures parity at "
    "each stage: evidence availability, retrieval, evidence coverage, dependence on high-resource sources, and the "
    "token cost of expressing the same need. We built 32 parallel information needs on education, employment, public "
    "services and digital safety, each with three verifiable key facts, and evaluated them in English, Turkish, Urdu, "
    "Swahili, Yoruba and Amharic against the live Wikipedia editions of those languages using a model-independent "
    "retrieval-augmented generation (RAG) retriever. "
    f"The canonical article for a need existed in {exists['tr']} of 32 cases for Turkish, {exists['ur']} for Urdu, "
    f"{exists['sw']} for Swahili, {exists['yo']} for Yoruba and {exists['am']} for Amharic. Native evidence covered "
    f"{pct(L.loc['en', 'native_cov'])} of key facts for English but {pct(low_native)} on average for the three "
    f"lowest-resource languages, while English-pivot evidence raised this to {pct(low_pivot)}. "
    f"When native and English evidence competed, English passages filled {pct(L.loc[low, 'mixed_en_share'].mean())} "
    "of top-ranked slots for these languages, and the same question cost up to "
    f"{L.loc[LANG_CODES, 'tok_qwen25'].max():.1f} times more tokens than in English. Information inequality is "
    "therefore already present before any text is generated, and closing it through English sources trades local "
    "relevance for coverage.",
    "multilingual large language models, information access, low-resource languages, retrieval-augmented generation, "
    "language equity")

# ---------------- 1. Introduction ----------------
b.heading("1. INTRODUCTION")
b.paragraph(
    "Large language models (LLMs) are increasingly used as an entry point to information about schooling, jobs, "
    "public services and online safety, and they are routinely presented as multilingual systems. The usual evidence "
    "for this claim comes from benchmarks of language understanding and generation, such as translation, reading "
    "comprehension and question answering scores [1-4]. These benchmarks show that performance falls with the amount "
    "of digital text available for a language [5, 6], and that tokenizers charge some languages several times more "
    "than others for the same content [7, 8]. They say much less about a more practical question: does a person who "
    "asks for help in Yoruba or Amharic receive substantively the same information as a person who asks in English?")
b.paragraph(
    "Deployed assistants rarely answer from parametric memory alone. They retrieve documents and condition their answer "
    "on that evidence, following the retrieval-augmented generation (RAG) paradigm [9]. In such systems, linguistic "
    "inequality can enter at several points: the model may misunderstand the question, the documents may not exist in "
    "the user's language, the retriever may fail to find them, or the generator may ignore or distort them. Recent "
    "work shows that multilingual RAG systems prefer evidence in the query language and fall back on high-resource "
    "languages when local content is scarce [10, 11], and that factual knowledge in LLMs is not consistent across "
    "languages [12]. What is missing is a framework that separates these failure sources, so that disparities can be "
    "attributed to a stage of the pipeline rather than to \"the model\" as a whole.")
b.paragraph(
    "This paper introduces LinguaParity, a framework for measuring information-access parity across languages, and "
    "reports a controlled study of its evidence layer. The contributions are: (i) a stage-wise definition of parity "
    "covering evidence availability, retrieval, evidence coverage, source-language dependence and token cost; "
    "(ii) a parallel set of 32 socially relevant information needs with verifiable key facts; (iii) a measurement of "
    "six languages, from English to Amharic, against the live Wikipedia editions that many RAG systems use as a "
    "knowledge source; and (iv) an open, reproducible implementation. The central finding is that much of the "
    "inequality is already in place before an LLM writes a single word.")

b.heading("1.1. Related Work", 2)
b.paragraph(
    "Joshi et al. [5] grouped the world's languages into resource classes and showed that most languages have almost "
    "no labelled data; Blasi et al. [6] linked performance gaps to the economic and demographic weight of language "
    "communities. Multilingual evaluations of generative models such as MEGA [3] confirm large drops for low-resource "
    "languages. Open-domain resources such as XOR QA [13], MKQA [14] and MIRACL [15] evaluate cross-lingual retrieval "
    "and answering, and Chirkova et al. [11] showed that multilingual RAG pipelines need careful choice of retriever "
    "and prompt language. Sharma et al. [10] found that LLMs in RAG settings favour information in the query language "
    "and otherwise default to high-resource sources, describing the result as a \"faux polyglot\" behaviour. Work on "
    "Wikipedia has long shown that language editions differ in content, not only in size [16]. LinguaParity builds on "
    "these strands but changes the unit of analysis from task accuracy to whether an equivalent information need can "
    "be served with equivalent evidence.")

# ---------------- 2. Materials and Methods ----------------
b.heading("2. MATERIALS AND METHODS")
b.heading("2.1. Information Needs and Languages", 2)
b.paragraph(
    "We wrote 32 information needs, eight in each of four domains: education (for example scholarships and distance "
    "education), employment (minimum wage, unemployment benefits, child labour), public services (passports, birth "
    "certificates, vaccination, voter registration) and digital safety (phishing, two-factor authentication, "
    "advance-fee scams). Each need is a natural question tied to one canonical English Wikipedia article and to "
    "three key facts stated in that article's introduction, which serves as the gold reference. Questions are general "
    "rather than country specific, so the same facts are correct in every language.")
b.paragraph(
    "Six languages span the resource classes of Joshi et al. [5]: English (class 5), Turkish (4), Urdu (3), and "
    "Swahili, Yoruba and Amharic (2). They cover Latin, Perso-Arabic and Ethiopic scripts. Questions were translated "
    "from English with NLLB-200 (distilled, 600M parameters) [17]. As a quality check, LaBSE [18] similarity between "
    f"each English question and its translation averaged between {L.loc[LANG_CODES[1:], 'mt_labse'].min():.2f} and "
    f"{L.loc[LANG_CODES[1:], 'mt_labse'].max():.2f}, and between {L.loc[LANG_CODES[1:], 'mt_roundtrip_labse'].min():.2f} "
    f"and {L.loc[LANG_CODES[1:], 'mt_roundtrip_labse'].max():.2f} after back-translation (Table 1). Machine-translated "
    "questions are a limitation that native-speaker review should remove in future versions.")

b.heading("2.2. Pipeline Stages and Measures", 2)
b.paragraph("For every need *q* and language *l*, LinguaParity measures the following stages.")
b.bullets([
    "Evidence availability: whether the language edition of Wikipedia contains the article linked to the canonical "
    "English article through interlanguage links, and its size relative to English.",
    "Retrieval: a model-independent retriever OR-combines the question tokens into a lexical query against the "
    f"edition's search engine, keeps the top 10 pages and reranks their introductions with LaBSE; the top 3 become "
    "evidence. Retrieval success is whether the canonical article is among them (hit@3).",
    "Evidence coverage: the share of key facts present in the evidence (Eq. 1). It is computed for native evidence, "
    "for English-pivot evidence (the question is machine-translated and English Wikipedia is searched) and for a "
    "mixed pool where native and English candidates are reranked together.",
    "Source-language dependence: the share of English passages in the mixed top 3.",
    "Token cost: the number of tokens the Qwen2.5 [19] and Qwen3 [20] tokenizer (the two share one vocabulary) needs "
    "for the same question, divided by the English count.",
])
b.paragraph(
    "Coverage is judged without a translation step. Each English key fact *f* and each evidence sentence *s* are "
    "embedded with LaBSE, a language-agnostic sentence encoder, and the coverage of need *q* in language *l* under "
    "evidence set *E* is")
b.equation(
    '<m:r><m:t>C</m:t></m:r><m:d><m:dPr><m:ctrlPr/></m:dPr><m:e><m:r><m:t>q,l,E</m:t></m:r></m:e></m:d>'
    '<m:r><m:t>=</m:t></m:r>'
    '<m:f><m:num><m:r><m:t>1</m:t></m:r></m:num><m:den><m:d><m:dPr><m:begChr m:val="|"/><m:endChr m:val="|"/></m:dPr>'
    '<m:e><m:sSub><m:e><m:r><m:t>F</m:t></m:r></m:e><m:sub><m:r><m:t>q</m:t></m:r></m:sub></m:sSub></m:e></m:d></m:den></m:f>'
    '<m:nary><m:naryPr><m:chr m:val="&#8721;"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/></m:naryPr>'
    '<m:sub><m:r><m:t>f&#8712;</m:t></m:r><m:sSub><m:e><m:r><m:t>F</m:t></m:r></m:e><m:sub><m:r><m:t>q</m:t></m:r></m:sub></m:sSub></m:sub><m:sup/>'
    '<m:e><m:r><m:t>&#120128;</m:t></m:r><m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr><m:e>'
    '<m:func><m:fName><m:limLow><m:e><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>max</m:t></m:r></m:e>'
    '<m:lim><m:r><m:t>s&#8712;E</m:t></m:r></m:lim></m:limLow></m:fName>'
    '<m:e><m:r><m:t>cos</m:t></m:r><m:d><m:dPr><m:ctrlPr/></m:dPr><m:e><m:r><m:t>e</m:t></m:r>'
    '<m:d><m:dPr><m:ctrlPr/></m:dPr><m:e><m:r><m:t>f</m:t></m:r></m:e></m:d><m:r><m:t>,</m:t></m:r>'
    '<m:r><m:t>e</m:t></m:r><m:d><m:dPr><m:ctrlPr/></m:dPr><m:e><m:r><m:t>s</m:t></m:r></m:e></m:d></m:e></m:d></m:e></m:func>'
    '<m:r><m:t>&#8805;&#964;</m:t></m:r></m:e></m:d></m:e></m:nary>', 1)
b.paragraph(
    "where *F*_{q} is the set of key facts, *e*(·) is the LaBSE embedding, 𝕀[·] is the indicator function and τ is a "
    "similarity threshold. τ was calibrated on English by maximising Youden's J between each fact and its own "
    "reference article versus an unrelated reference article "
    f"(τ = {cal['tau']:.3f}, AUC = {cal['auc']:.3f}, true positive rate {cal['tpr']:.2f} at false positive rate "
    f"{cal['fpr']:.2f}, {cal['n_pairs']} pairs). Because the negatives are unrelated articles, this calibration "
    "separates on-topic from off-topic text and is deliberately lenient; coverage values are therefore upper bounds.")
b.paragraph(
    "Paired differences across the 32 needs were tested with the Wilcoxon signed-rank test and Holm correction over "
    "the five non-English languages; 95% confidence intervals are percentile bootstraps over needs (5,000 resamples). "
    "All Wikipedia responses were cached, so every number in this paper can be recomputed from the released snapshot. "
    "Everything ran on a CPU-only laptop. The code and data are available at github.com/inayatarshad/linguaparity.")

# ---------------- 3. Results ----------------
b.heading("3. RESULTS AND DISCUSSION")
b.heading("3.1. Evidence Availability and Retrieval", 2)
b.paragraph(
    "Table 1 summarises the pipeline for each language. Availability falls steeply with resource level: the canonical "
    f"article existed for all 32 needs in English, {exists['tr']} in Turkish, {exists['ur']} in Urdu, {exists['sw']} "
    f"in Swahili, {exists['yo']} in Yoruba and {exists['am']} in Amharic. Where articles existed they were much shorter "
    f"than their English counterparts (mean size ratio {f2(L.loc['tr', 'bytes_ratio'])} for Turkish and "
    f"{f2(L.loc['ur', 'bytes_ratio'])} for Urdu, counting missing articles as zero). Raw edition size is a poor proxy "
    f"for availability: Urdu Wikipedia has about as many articles as Turkish ({L.loc['ur', 'wiki_articles']:,} against "
    f"{L.loc['tr', 'wiki_articles']:,}) yet covers far fewer of these needs, which suggests that many of its pages do "
    "not address everyday social topics. Retrieval tracks availability: the canonical article reached the native top 3 "
    f"for {pct(L.loc['en', 'native_hit3'])} of English needs, {pct(L.loc['tr', 'native_hit3'])} of Turkish, "
    f"{pct(L.loc['ur', 'native_hit3'])} of Urdu and {pct(L.loc['sw', 'native_hit3'])} of Swahili needs.")

b.caption("Table 1.", "Stage-wise results per language (32 needs). Cov. = key-fact coverage of top-3 evidence; "
          "EN share = English passages in mixed top 3; Tok. = Qwen2.5 tokens relative to English.")
hdr = ["Language", "Class", "Wiki articles", "Article exists", "Hit@3", "MT LaBSE", "Native cov.", "Pivot cov.",
       "Mixed cov.", "EN share", "Tok."]
rows = [hdr]
for l in LANG_CODES:
    r = L.loc[l]
    rows.append([r["name"], int(r["joshi"]), f"{int(r['wiki_articles']):,}", f2(r["article_exists"]),
                 f2(r["native_hit3"]), f2(r["mt_labse"]), f2(r["native_cov"]), f2(r["pivot_cov"]),
                 f2(r["mixed_cov"]), "-" if l == "en" else f2(r["mixed_en_share"]), f"{r['tok_qwen25']:.2f}"])
b.table(rows, [1.7, 1.0, 1.7, 1.4, 1.2, 1.4, 1.4, 1.4, 1.4, 1.3, 1.1], font_half_points=14)
b.spacer()

b.heading("3.2. Evidence Coverage and Dependence on English", 2)
b.paragraph(
    "Figure 1 follows each language through the pipeline. Native evidence covered "
    f"{pct(L.loc['en', 'native_cov'])} of key facts in English, compared with "
    + ", ".join(f"{pct(L.loc[l, 'native_cov'])} in {L.loc[l, 'name']}" for l in LANG_CODES[1:])
    + ". "
    + " ".join(
        f"The gap to English was significant for {', '.join(L.loc[l, 'name'] for l in LANG_CODES[1:] if nve[l]['p_holm'] < 0.05)} "
        "(Wilcoxon, Holm-corrected p < 0.05)." for _ in [0])
    + f" At the level of individual needs, whether a native article existed was associated with native coverage "
    f"(Spearman ρ = {sp['article_exists']['rho']:.2f}, {p_fmt(sp['article_exists']['p'])}, n = {sp['article_exists']['n']}).")
b.figure(FIGURES / "fig1_pipeline.png", 15.5)
b.caption("Figure 1.", "Information-access pipeline per language: availability of the canonical article, retrieval "
          "success, key-fact coverage of native and English-pivot evidence, and the English share of a mixed ranking.")
b.paragraph(
    "Routing the question through English closes much of this gap. English-pivot evidence covered "
    + ", ".join(f"{pct(pvn[l]['pivot'])} for {L.loc[l, 'name']}" for l in LANG_CODES[1:])
    + f"; the gain over native evidence averaged {pct(low_pivot - low_native)} for the three lowest-resource "
    "languages. When native and English candidates were ranked together, English passages took "
    + ", ".join(f"{pct(L.loc[l, 'mixed_en_share'])} of slots for {L.loc[l, 'name']}" for l in LANG_CODES[1:])
    + ". In other words, a multilingual RAG system that serves these users well does so by quietly answering from "
    "English sources. This agrees with the high-resource fallback reported by Sharma et al. [10], and it carries a "
    "cost that coverage scores do not capture: English Wikipedia describes passports, pensions or unemployment "
    "benefits mainly from the perspective of high-income countries, so the answer can be complete but locally "
    "irrelevant. The pivot also depends on machine translation of the question, which is itself weakest for the "
    "same languages (Table 1).")

b.heading("3.3. Domains and Token Cost", 2)
b.paragraph(
    "Figure 2 breaks native coverage down by domain. Public services were the best covered domain in Urdu, Swahili, "
    "Yoruba and Amharic, while digital safety and employment were the weakest: for the three lowest-resource "
    f"languages, native evidence covered on average {pct(sum(dom[l]['digital_safety'] for l in low) / 3)} of "
    f"digital-safety facts and {pct(sum(dom[l]['employment'] for l in low) / 3)} of employment facts. These are the "
    "domains where new internet users most need timely, local guidance on scams, labour rights and benefits. "
    "Expressing the same question also cost more: relative to English, the tokenizer shared by Qwen2.5 and Qwen3 "
    "needed "
    + ", ".join(f"{L.loc[l, 'tok_qwen25']:.1f}x as many tokens for {L.loc[l, 'name']}" for l in LANG_CODES[1:])
    + ". Longer token sequences mean higher cost per query, less room for evidence within a fixed context window "
    "and slower responses, which compounds the evidence gap [7, 8].")
b.figure(FIGURES / "fig2_domains.png", 15.5)
b.caption("Figure 2.", "Key-fact coverage of native top-3 evidence by domain and language (8 needs per cell).")

b.heading("3.4. Limitations", 2)
b.paragraph(
    "This study measures the evidence layer of the pipeline; the generation layer of the LinguaParity framework "
    "(answer completeness, factual consistency, citation quality and hallucination) is specified but not yet "
    "reported, because running LLMs on CPU was beyond the compute available for this version. The evidence results "
    "nonetheless bound what any RAG-based answer can contain. Questions were machine translated, Wikipedia is only "
    "one possible knowledge source, the needs are general rather than country specific, and LaBSE similarity is a "
    "proxy for factual entailment whose reliability is itself likely to be lower for low-resource languages. "
    "The study covers six languages and 32 needs, so language-level correlations are indicative only.")

# ---------------- 4. Conclusion ----------------
b.heading("4. CONCLUSION")
b.paragraph(
    "LinguaParity reframes multilingual AI evaluation as a question of information access. Across six languages and "
    "32 socially relevant needs, the evidence that a retrieval-augmented assistant could draw on shrank sharply with "
    "resource level, from full availability in English to almost none in Yoruba and Amharic, and retrieval and "
    "coverage followed. English sources closed much of the gap, but only by making low-resource users dependent on "
    "content written for other contexts, while tokenization made each of their questions more expensive. Because "
    "these disparities arise before generation, improving the language skills of LLMs alone will not equalise access. "
    "Investment in local-language content on public services, employment, education and online safety, in "
    "multilingual retrieval, and in evaluation of locally relevant answers is needed. Future work will add the "
    "generation-stage measurements with several open and commercial LLMs, native-speaker validation of the needs and "
    "country-specific information needs.")

b.heading("ACKNOWLEDGMENT")
b.paragraph("The authors thank the Wikipedia communities whose open content made this measurement possible.")
b.heading("DECLARATIONS")
b.paragraph(
    "Funding: none. Conflict of interest: the authors declare no conflict of interest. Ethics: the study uses only "
    "public encyclopedic content and involves no human participants. Data availability: the information needs, "
    "cached Wikipedia responses, code and results are available at https://github.com/inayatarshad/linguaparity.")

b.heading("REFERENCES")
b.references([
    "NLLB Team, M. R. Costa-jussà et al., \"No language left behind: Scaling human-centered machine translation,\" "
    "arXiv:2207.04672, 2022.",
    "L. Bandarkar et al., \"The Belebele benchmark: A parallel reading comprehension dataset in 122 language "
    "variants,\" in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics, Bangkok, Thailand, 2024, pp. 749-775.",
    "K. Ahuja et al., \"MEGA: Multilingual evaluation of generative AI,\" in Proc. Conf. Empirical Methods Natural "
    "Language Process., Singapore, 2023, pp. 4232-4267.",
    "V. D. Lai et al., \"ChatGPT beyond English: Towards a comprehensive evaluation of large language models in "
    "multilingual learning,\" in Findings Assoc. Comput. Linguistics: EMNLP 2023, Singapore, 2023.",
    "P. Joshi, S. Santy, A. Budhiraja, K. Bali, and M. Choudhury, \"The state and fate of linguistic diversity and "
    "inclusion in the NLP world,\" in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics, 2020, pp. 6282-6293.",
    "D. Blasi, A. Anastasopoulos, and G. Neubig, \"Systematic inequalities in language technology performance across "
    "the world's languages,\" in Proc. 60th Annu. Meeting Assoc. Comput. Linguistics, Dublin, Ireland, 2022, "
    "pp. 5486-5505.",
    "O. Ahia et al., \"Do all languages cost the same? Tokenization in the era of commercial language models,\" in "
    "Proc. Conf. Empirical Methods Natural Language Process., Singapore, 2023, pp. 9904-9923.",
    "A. Petrov, E. La Malfa, P. H. S. Torr, and A. Bibi, \"Language model tokenizers introduce unfairness between "
    "languages,\" in Advances Neural Inf. Process. Syst., vol. 36, 2023.",
    "P. Lewis et al., \"Retrieval-augmented generation for knowledge-intensive NLP tasks,\" in Advances Neural Inf. "
    "Process. Syst., vol. 33, 2020, pp. 9459-9474.",
    "N. Sharma, K. Murray, and Z. Xiao, \"Faux polyglot: A study on information disparity in multilingual large "
    "language models,\" in Proc. Conf. Nations Americas Chapter Assoc. Comput. Linguistics, Albuquerque, NM, USA, "
    "2025, pp. 8090-8107.",
    "N. Chirkova, D. Rau, H. Déjean, T. Formal, S. Clinchant, and V. Nikoulina, \"Retrieval-augmented generation in "
    "multilingual settings,\" in Proc. 1st Workshop Towards Knowledgeable Language Models, Bangkok, Thailand, 2024, "
    "pp. 177-188.",
    "J. Qi, R. Fernández, and A. Bisazza, \"Cross-lingual consistency of factual knowledge in multilingual language "
    "models,\" in Proc. Conf. Empirical Methods Natural Language Process., Singapore, 2023, pp. 10650-10666.",
    "A. Asai, J. Kasai, J. H. Clark, K. Lee, E. Choi, and H. Hajishirzi, \"XOR QA: Cross-lingual open-retrieval "
    "question answering,\" in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics, 2021, pp. 547-564.",
    "S. Longpre, Y. Lu, and J. Daiber, \"MKQA: A linguistically diverse benchmark for multilingual open domain "
    "question answering,\" Trans. Assoc. Comput. Linguistics, vol. 9, pp. 1389-1406, 2021.",
    "X. Zhang et al., \"MIRACL: A multilingual retrieval dataset covering 18 diverse languages,\" Trans. Assoc. "
    "Comput. Linguistics, vol. 11, pp. 1114-1131, 2023.",
    "B. Hecht and D. Gergle, \"The tower of Babel meets Web 2.0: User-generated content and its applications in a "
    "multilingual context,\" in Proc. SIGCHI Conf. Human Factors Comput. Syst., Atlanta, GA, USA, 2010, pp. 291-300.",
    "NLLB Team, \"Scaling neural machine translation to 200 languages,\" Nature, vol. 630, pp. 841-846, 2024.",
    "F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, \"Language-agnostic BERT sentence embedding,\" in Proc. "
    "60th Annu. Meeting Assoc. Comput. Linguistics, Dublin, Ireland, 2022, pp. 878-891.",
    "Qwen Team, \"Qwen2.5 technical report,\" arXiv:2412.15115, 2024.",
    "Qwen Team, \"Qwen3 technical report,\" arXiv:2505.09388, 2025.",
])
b.save(out_path)
print("saved", out_path)
