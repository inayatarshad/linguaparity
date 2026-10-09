"""Minimal cached MediaWiki API client.

Every response is cached on disk so that a run can be repeated against the
same snapshot of Wikipedia.
"""
import hashlib
import json
import time

import requests

from .config import CACHE

_S = requests.Session()
_S.headers["User-Agent"] = (
    "LinguaParity-research/0.1 (https://github.com/inayatarshad/linguaparity)"
)


def api(lang, **params):
    params = {"action": "query", "format": "json", "formatversion": 2, **params}
    key = hashlib.sha1(json.dumps([lang, params], sort_keys=True).encode()).hexdigest()
    path = CACHE / "wiki" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf8"))
    for attempt in range(5):
        try:
            r = _S.get(f"https://{lang}.wikipedia.org/w/api.php", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError):
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"MediaWiki API failed for {lang} {params}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
    time.sleep(0.1)
    return data


def article_count(lang):
    d = api(lang, meta="siteinfo", siprop="statistics")
    return d["query"]["statistics"]["articles"]


def page_info(lang, title):
    """Intro extract, byte length and interlanguage links of one page."""
    d = api(lang, titles=title, prop="extracts|info|langlinks", exintro=1,
            explaintext=1, lllimit=500, redirects=1)
    p = d["query"]["pages"][0]
    if p.get("missing"):
        return None
    return {
        "title": p["title"],
        "length": p.get("length", 0),
        "extract": p.get("extract", ""),
        "langlinks": {x["lang"]: x["title"] for x in p.get("langlinks", [])},
    }


def search(lang, query, limit):
    d = api(lang, list="search", srsearch=query, srlimit=limit, srwhat="text")
    return [x["title"] for x in d["query"]["search"]]


def intros(lang, titles):
    """Intro extracts for up to 20 titles (one API call)."""
    if not titles:
        return {}
    d = api(lang, titles="|".join(titles), prop="extracts", exintro=1,
            explaintext=1, exlimit="max", redirects=1)
    return {p["title"]: p.get("extract", "") for p in d["query"]["pages"]
            if not p.get("missing")}
