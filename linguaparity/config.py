"""Shared configuration: languages, model ids, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = DATA / "cache"

# Languages ordered from high to low resource. `joshi` is the resource class
# from Joshi et al. (2020), 5 = winners ... 0 = left-behinds.
LANGUAGES = {
    "en": {"name": "English", "nllb": "eng_Latn", "joshi": 5},
    "tr": {"name": "Turkish", "nllb": "tur_Latn", "joshi": 4},
    "ur": {"name": "Urdu", "nllb": "urd_Arab", "joshi": 3},
    "sw": {"name": "Swahili", "nllb": "swh_Latn", "joshi": 2},
    "yo": {"name": "Yoruba", "nllb": "yor_Latn", "joshi": 2},
    "am": {"name": "Amharic", "nllb": "amh_Ethi", "joshi": 2},
}
LANG_CODES = list(LANGUAGES)

MT_MODEL = "facebook/nllb-200-distilled-600M"
EMBED_MODEL = "sentence-transformers/LaBSE"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
GENERATORS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

SEED = 13
TOP_K = 3            # evidence passages given to the generator
CANDIDATES = 20      # candidate pages pulled from search before dense reranking
PASSAGE_CHARS = 700  # characters kept per evidence passage
MAX_NEW_TOKENS = 160


def load_needs():
    import json
    return json.loads((DATA / "information_needs.json").read_text(encoding="utf8"))
