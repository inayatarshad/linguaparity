import re

# Sentence-final punctuation incl. Urdu (U+06D4, U+061F) and Ethiopic (U+1362, U+1367).
_SPLIT = re.compile(r"(?<=[.!?\u06d4\u061f\u1362\u1367])\s+|\n+")


def sentences(text, min_chars=3):
    return [s.strip() for s in _SPLIT.split(text or "") if len(s.strip()) >= min_chars]


def windows(sents, size=3):
    """Overlapping windows of consecutive sentences, used as NLI premises."""
    if len(sents) <= size:
        return [" ".join(sents)] if sents else [""]
    return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]
