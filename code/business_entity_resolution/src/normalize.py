"""Text normalization utilities for business names and addresses.

All functions are pure string -> string / string -> list transforms so they can be
mapped in bulk over a Polars column (via ``map_elements``) or called row-wise.
No external data or network lookups are used anywhere in this module.
"""
import re
import unicodedata

# Legal-entity / trade-form tokens that carry ~no discriminative signal once a
# business name has been tokenized. Stripped when building the "core name" used
# for blocking keys and one of the similarity features (raw name is kept as a
# separate feature so genuine differences are not thrown away).
LEGAL_SUFFIX_TOKENS = {
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "llp", "plc", "pc", "lp", "dba", "gmbh",
    "sa", "sas", "srl", "bv", "nv", "the",
}

# Generic address words that appear in almost every record and would blow up an
# inverted index if used as blocking keys. They are still used in similarity
# features (via the normalized full string), just excluded from blocking tokens.
ADDRESS_STOPWORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "drive", "dr", "lane", "ln",
    "boulevard", "blvd", "court", "ct", "circle", "cir", "way", "place", "pl",
    "near", "opp", "opposite", "behind", "unit", "suite", "ste", "apt",
    "apartment", "floor", "bldg", "building", "block", "colony", "nagar",
    "sector", "phase", "po", "box", "pmb", "no", "number", "and", "of", "the",
    "in", "at",
}

ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "dr": "drive", "ln": "lane",
    "blvd": "boulevard", "ct": "court", "cir": "circle", "apt": "apartment",
    "ste": "suite", "bldg": "building", "hwy": "highway", "pkwy": "parkway",
    "sq": "square", "mt": "mount", "ft": "fort", "co": "county",
}

# Visually-confusable digit -> letter substitutions ("leet speak"), used to undo
# the character-corruption noise pattern seen in the data (e.g. "vá1ley" ->
# "valley", "Aut0" -> "auto", "5hreeman" -> "shreeman").
LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})

_DOMAIN_TLD_RE = re.compile(r"\.(com|net|org|in|co|biz|info|us|io)\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def strip_accents(text: str) -> str:
    if text is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def basic_clean(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace. Keeps non-Latin scripts intact."""
    if text is None:
        return ""
    text = strip_accents(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_name(raw: str) -> str:
    text = basic_clean(raw)
    text = _DOMAIN_TLD_RE.sub(" ", text)
    text = text.replace("www ", " ")
    return _WS_RE.sub(" ", text).strip()


def leet_fix(text: str) -> str:
    """Undo digit-for-letter corruption; only applied to alnum tokens with letters."""
    out_tokens = []
    for tok in text.split(" "):
        if tok and any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            out_tokens.append(tok.translate(LEET_MAP))
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def core_name_tokens(cleaned_name: str) -> list:
    """Tokens of a cleaned name with legal-suffix / filler tokens removed."""
    toks = leet_fix(cleaned_name).split(" ")
    return [t for t in toks if t and t not in LEGAL_SUFFIX_TOKENS and len(t) >= 2]


def squeeze(text: str) -> str:
    """Alphanumeric-only, space-free string; robust to spacing/punctuation noise
    and to domain-name-style concatenation of the business name."""
    return _NON_ALNUM_RE.sub("", leet_fix(text))


def char_ngrams(text: str, n: int = 3) -> set:
    s = squeeze(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def name_char_ngrams_for_blocking(cleaned_name: str, n: int = 5) -> list:
    """5-grams over the squeezed (space/punctuation-free) name, used as a blocking
    channel that survives typos/corruption inside individual word tokens (the
    token-based name channel requires a whole word to match intact)."""
    return list(char_ngrams(cleaned_name, n=n))


def clean_address(raw: str) -> str:
    if raw is None:
        return ""
    text = basic_clean(raw)
    toks = [ADDRESS_ABBREV.get(t, t) for t in text.split(" ")]
    return " ".join(t for t in toks if t)


def address_tokens_for_blocking(cleaned_addr: str) -> list:
    toks = leet_fix(cleaned_addr).split(" ")
    return [t for t in toks if t and t not in ADDRESS_STOPWORDS and len(t) >= 3]


def digit_tokens(cleaned_addr: str) -> list:
    return _DIGIT_RE.findall(cleaned_addr)


def digit_tokens_for_blocking(cleaned_addr: str) -> list:
    """Digit blocking-channel keys: same digit runs as digit_tokens(), but
    requiring length >= 3, matching the length floor every other blocking
    extractor already applies (name/address tokens require >= 2-3 chars).
    Short digit runs (single/double-digit street numbers) are extremely
    common and carry almost no discriminative signal, yet unfiltered they
    have document frequency in the tens of thousands — large enough to blow
    up the blocking join's fan-out. This filters them for blocking KEYS only;
    the digit_jaccard/digit_overlap_count features still use the unfiltered
    digit_tokens() since a short shared digit is still weak-but-real evidence
    once two records are already candidates."""
    return [t for t in digit_tokens(cleaned_addr) if len(t) >= 3]


def name_blocking_tokens(cleaned_name: str) -> list:
    """Tokens used as blocking keys for the name channel: core tokens, length-filtered."""
    return [t for t in core_name_tokens(cleaned_name) if len(t) >= 3]
