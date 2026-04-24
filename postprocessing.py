"""
Post-processing module for OCR output.
Corrects common OCR mistakes, normalises spacing, preserves document structure.
"""

import re
from difflib import get_close_matches

# ─── Word-level corrections ──────────────────────────────────────────────────
# Applied as whole-word regex substitutions (case-sensitive).
_WORD_FIXES: list[tuple[str, str]] = [
    # ── Acronyms / domain terms (banking / finance) ──────────────────────
    ("Cibil",  "CIBIL"),  ("cibil",  "CIBIL"),
    ("Kyc",    "KYC"),    ("kyc",    "KYC"),
    ("LAp",    "LAP"),    ("Lap",    "LAP"),
    ("Emi",    "EMI"),    ("emi",    "EMI"),
    ("Noc",    "NOC"),    ("noc",    "NOC"),
    ("Roi",    "ROI"),
    ("Upi",    "UPI"),    ("upi",    "UPI"),
    ("Imps",   "IMPS"),   ("imps",   "IMPS"),
    ("Neft",   "NEFT"),   ("neft",   "NEFT"),
    ("Rtgs",   "RTGS"),   ("rtgs",   "RTGS"),
    # ── Computer / Network domain terms ─────────────────────────────────────
    ("Mehxopolitan",      "Metropolitan"),    ("mehxopolitan",      "metropolitan"),
    ("Netwosk",           "Network"),         ("netwosk",           "network"),
    ("Netwerk",           "Network"),         ("netwerk",           "network"),
    ("Netwouk",           "Network"),         ("netwouk",           "network"),
    ("Compufer",          "Computer"),        ("compufer",          "computer"),
    ("Compuler",          "Computer"),        ("compuler",          "computer"),
    ("Computrs",          "Computers"),       ("computrs",          "computers"),
    ("Compufews",         "Computers"),       ("compufews",         "computers"),
    ("Hhglobe",           "globe"),           ("hhglobe",           "globe"),
    ("Geographico",       "Geographic"),      ("geographico",       "geographic"),
    ("Couened",           "covered"),         ("couened",           "covered"),
    ("Tuaose",            "those"),           ("tuaose",            "those"),
    ("Millitrs",          "millions"),        ("millitrs",          "millions"),
    ("Spuegd",            "spread"),          ("spuegd",            "spread"),
    ("Connecteq",         "connected"),       ("connecteq",         "connected"),
    ("Thaoygh",           "through"),         ("thaoygh",           "through"),
    ("Bluetoom",          "Bluetooth"),       ("bluetoom",          "bluetooth"),
    ("Handbeld",          "handheld"),        ("handbeld",          "handheld"),
    # ── Short common misreads (4–5 chars) ────────────────────────────────────
    ("Auea",              "Area"),            ("auea",              "area"),
    ("Aea",               "Area"),            ("aea",               "area"),
    ("Singe",             "Since"),           ("singe",             "since"),
    ("Acoss",             "Across"),          ("acoss",             "across"),
    ("Dafa",              "Data"),            ("dafa",              "data"),
    ("Oum",               "Our"),             ("oum",               "our"),
    ("Oom",               "on"),              ("oom",               "on"),
    # ── Merged-word TrOCR artifacts (two words run together) ─────────────────
    ("connecteqthaoygh",  "connected through"),("Connecteqthaoygh",  "Connected through"),
    ("Singe oom",         "Since on"),
    ("Acoss tablets",     "Across tablets"),
    # ── Common OCR mis-recognitions ──────────────────────────────────────
    ("Scoe",         "Score"),       ("scoe",         "score"),
    ("Scove",        "Score"),
    ("Whhat",        "What"),        ("INhat",        "What"),  ("Wbat", "What"),
    ("Arpoved",      "Approved"),    ("arpoved",      "approved"),
    ("Pre-Arpoved",  "Pre-Approved"),("pre-arpoved",  "pre-approved"),
    ("documerts",    "documents"),   ("documenis",    "documents"),
    ("Documerts",    "Documents"),   ("Documenis",    "Documents"),
    ("numbrer",      "number"),      ("Numbrer",      "Number"),
    ("cedit",        "credit"),      ("Cedit",        "Credit"),
    ("loar",         "loan"),        ("Loar",         "Loan"),
    ("applicarion",  "application"), ("Applicarion",  "Application"),
    ("signaiure",    "signature"),   ("Signaiure",    "Signature"),
    ("verifiacion",  "verification"),("Verifiacion",  "Verification"),
    ("approvel",     "approval"),    ("Approvel",     "Approval"),
    ("acount",       "account"),     ("Acount",       "Account"),
    ("banck",        "bank"),        ("Banck",        "Bank"),
]

# ─── Regex-based pattern corrections ────────────────────────────────────────
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Digit 'l' at end of number: 9l → 91
    (re.compile(r'(?<=\d)l\b'), '1'),
    # Letter O surrounded by digits: 1O5 → 105
    (re.compile(r'(?<=\d)O(?=\d)'), '0'),
    # Punctuation glued to next word: "text,word" → "text, word"
    (re.compile(r'([.,;:!?])([A-Za-z])'), r'\1 \2'),
    # Multiple consecutive dots normalised to ellipsis
    (re.compile(r'\.{4,}'), '...'),
    # Stray lone hyphen on its own line (OCR artifact)
    (re.compile(r'(?m)^-\s*$'), ''),
    # Double spaces inside a word (OCR splits character): "d ocument" → "document"  (NOT done globally — too risky)
    # Fix "Rs ." → "Rs.", "No ." → "No."
    (re.compile(r'(\w)\s\.'), r'\1.'),
]

# ─── Known acronyms (financial + network) for uppercase normalisation ────────
_ACRONYMS: frozenset[str] = frozenset({
    "cibil", "kyc", "lap", "emi", "pan", "gst", "noc", "roi",
    "upi", "imps", "neft", "rtgs", "atm", "ocr", "apr",
    # Network / computer
    "lan", "wan", "pan", "man", "wlan", "vpn", "tcp", "ip",
    "http", "https", "ftp", "dns", "dhcp", "mac", "osi",
    "cpu", "ram", "rom", "gpu", "usb", "hdmi", "ssd", "hdd",
    "iot", "ai", "ml", "api", "url", "html", "css", "sql",
})

# ─── Domain vocabulary for fuzzy correction ──────────────────────────────────
_DOMAIN_VOCAB: list[str] = [
    # Network
    "network", "metropolitan", "internet", "wireless", "bluetooth",
    "connection", "transmission", "bandwidth", "protocol", "router",
    "gateway", "firewall", "server", "client", "switch", "interface",
    # Computer
    "computer", "computers", "hardware", "software", "processor",
    "database", "algorithm", "digital", "electronic", "device",
    "system", "technology", "communication", "information", "processing",
    # Short common words (added for 4-char fuzzy matching)
    "area", "data", "local", "wide", "personal", "since", "across",
    "covered", "those", "through", "spread", "millions", "globe",
    # Geography / general academic
    "geographic", "geographical", "international", "organization",
    "structure", "connected", "handheld", "portable", "mobile",
    "definition", "classification", "generation", "collection",
    "application", "installation", "configuration", "performance", "security",
]
_DOMAIN_VOCAB_LOWER: list[str] = [v.lower() for v in _DOMAIN_VOCAB]

# ─── Phrase-level corrections ─────────────────────────────────────────────────
# Applied AFTER word-level corrections. Catches multi-word patterns that
# individual word fixes cannot reconstruct (e.g. garbled "X Area Network").
_PHRASE_FIXES: list[tuple[re.Pattern, str]] = [
    # Network type phrases — match common TrOCR misreads of each word
    (re.compile(
        r'\b(?:Metropolitan|Mehxopolitan|Metr\w+)\s+(?:Area|Auea|Aea)\s+(?:Network|Netw\w+)\b',
        re.I), 'Metropolitan Area Network'),
    (re.compile(
        r'\bLocal\s+(?:Area|Auea|Aea)\s+(?:Network|Netw\w+)\b',
        re.I), 'Local Area Network'),
    (re.compile(
        r'\bWide\s+(?:Area|Auea|Aea)\s+(?:Network|Netw\w+)\b',
        re.I), 'Wide Area Network'),
    (re.compile(
        r'\bPersonal\s+(?:Area|Auea|Aea)\s+(?:Network|Netw\w+)\b',
        re.I), 'Personal Area Network'),
    (re.compile(
        r'\bWireless\s+(?:Local\s+)?(?:Area|Auea|Aea)\s+(?:Network|Netw\w+)\b',
        re.I), 'Wireless Local Area Network'),
    # Computer / device phrases
    (re.compile(r'\bPersonal\s+(?:Computer|Compufer|Compuler)\b', re.I), 'Personal Computer'),
    (re.compile(r'\bOperating\s+(?:System|Systm)\b', re.I), 'Operating System'),
    (re.compile(r'\bWi[-\s]?fi\b', re.I), 'Wi-Fi'),
]


def correct_phrases(text: str) -> str:
    """Apply phrase-level corrections (multi-word patterns)."""
    for pattern, replacement in _PHRASE_FIXES:
        text = pattern.sub(replacement, text)
    return text


def fuzzy_domain_correct(text: str, cutoff: float = 0.78) -> str:
    """
    Fuzzy-match words against domain vocabulary and correct close misreads.
    Processes words of 4+ characters (lowered from 6 to catch short misreads
    like 'auea'→'area', 'dafa'→'data').
    Uses difflib.get_close_matches (no extra dependencies).
    """
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        tokens = line.split(" ")
        corrected: list[str] = []
        for tok in tokens:
            alpha = re.sub(r"[^a-zA-Z]", "", tok)
            if len(alpha) < 4 or tok.isupper() or alpha.upper() in {a.upper() for a in _ACRONYMS}:
                corrected.append(tok)
                continue
            matches = get_close_matches(alpha.lower(), _DOMAIN_VOCAB_LOWER, n=1, cutoff=cutoff)
            if matches:
                best = next((v for v in _DOMAIN_VOCAB if v.lower() == matches[0]), matches[0])
                if alpha[0].isupper():
                    best = best.capitalize()
                tok = tok.replace(alpha, best, 1)
            corrected.append(tok)
        result.append(" ".join(corrected))
    return "\n".join(result)


def compute_ocr_quality(text: str) -> int:
    """
    Returns a quality score 0–100 for the OCR output.
    - 70-100: clean, mostly known words
    - 40-69 : some noise, usable
    - 0-39  : heavy garbage, likely needs manual review
    """
    if not text or not text.strip():
        return 0
    words = re.findall(r"[a-zA-Z]{3,}", text)
    if not words:
        return 5
    try:
        from spellchecker import SpellChecker  # type: ignore
        spell = SpellChecker()
        unknown = spell.unknown(words)
        known_ratio = 1.0 - (len(unknown) / len(words))
    except ImportError:
        known_ratio = 0.65  # fallback if pyspellchecker missing
    # Penalty for known garbage patterns
    garbage = len(re.findall(r"\b\d\s\d\b", text))        # "0 0" style
    garbage += len(re.findall(r"\b[a-z]{1}\b", text)) // 3  # single chars
    score = int(known_ratio * 100) - min(garbage * 3, 25)
    return max(0, min(100, score))


def correct_text(text: str) -> str:
    """Apply word-level and pattern-based corrections."""
    if not text:
        return text

    # Word-level (whole-word, case-sensitive)
    for wrong, right in _WORD_FIXES:
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text)

    # Regex patterns
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)

    return text


def normalize_whitespace(text: str) -> str:
    """Collapse inline spaces/tabs; allow at most one consecutive blank line."""
    lines = text.split('\n')
    result: list[str] = []
    blank_run = 0
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).rstrip()
        if line == '':
            blank_run += 1
            if blank_run <= 1:
                result.append('')
        else:
            blank_run = 0
            result.append(line)
    return '\n'.join(result).strip()


def fix_numbered_list(text: str) -> str:
    """
    Capitalise the first letter of content after a numbered list marker.
    e.g.  "1) what" → "1) What" ,  "2.check" → "2. Check"
    Works together with fix_merged_list_items (which adds the space first).
    """
    return re.sub(
        r'(\d+[.):])\s*([a-z])',
        lambda m: f"{m.group(1)} {m.group(2).upper()}",
        text,
    )


def fix_merged_list_items(text: str) -> str:
    """
    Fix list-number artifacts produced by OCR:
    - "1)word"  → "1) word"  (no space after closer)
    - "2. word" is already fine
    - "3)Word"  → "3) Word"
    """
    lines = text.split('\n')
    fixed = []
    for line in lines:
        # Ensure space after list marker: "1)" / "1." / "1:"
        line = re.sub(r'^(\d+[.):])(\S)', r'\1 \2', line.strip())
        fixed.append(line)
    return '\n'.join(fixed)


def normalize_acronyms(text: str) -> str:
    """
    Uppercase known financial / domain acronyms that appear in wrong case.
    Only matches exact whole-word occurrences.
    """
    def _upper_if_known(m: re.Match) -> str:
        w = m.group(0)
        return w.upper() if w.lower() in _ACRONYMS else w

    return re.sub(r'\b[A-Za-z]{2,5}\b', _upper_if_known, text)


def restore_structure(text: str) -> str:
    """
    Normalise list formatting detected in the text:
    - "1." / "1)" / "1:" → "1. <content>"
    - "*" / "-" / "•"   → "  • <content>"
    Page separators ("--- Page N ---") are preserved as-is.
    """
    lines = text.split('\n')
    out: list[str] = []
    for line in lines:
        stripped = line.strip()

        if not stripped:
            out.append('')
            continue

        # Page separators – keep unchanged
        if re.match(r'^---\s+Page\s+\d+', stripped):
            out.append(stripped)
            continue

        # Numbered list: "1." / "1)" / "1:"
        m = re.match(r'^(\d+)[.):\-]\s+(.+)$', stripped)
        if m:
            out.append(f"{m.group(1)}. {m.group(2)}")
            continue

        # Bullet point: "* text" / "- text" / "• text"
        m = re.match(r'^[*\-•]\s+(.+)$', stripped)
        if m:
            out.append(f"  • {m.group(1)}")
            continue

        out.append(stripped)

    return '\n'.join(out)


def word_level_correct(text: str) -> str:
    """
    Word-level spell correction using pyspellchecker.
    Corrects likely OCR word-read errors while protecting:
    - Acronyms (CIBIL, KYC …)
    - ALL-CAPS words
    - Numbers and tokens shorter than 4 alpha chars
    Falls back silently if pyspellchecker is not installed.
    """
    try:
        from spellchecker import SpellChecker  # type: ignore
    except ImportError:
        return text

    spell = SpellChecker()
    _protected = {a.upper() for a in _ACRONYMS}
    lines = text.split('\n')
    out: list[str] = []

    for line in lines:
        tokens = line.split(' ')
        corrected: list[str] = []
        for tok in tokens:
            alpha = re.sub(r'[^a-zA-Z]', '', tok)
            if (
                len(alpha) < 4          # too short to correct reliably
                or tok.isupper()        # all-caps → likely acronym
                or alpha.upper() in _protected
                or not alpha
            ):
                corrected.append(tok)
                continue

            fix = spell.correction(alpha.lower())
            if fix and fix != alpha.lower():
                # Preserve leading capitalisation
                if alpha[0].isupper():
                    fix = fix.capitalize()
                tok = tok.replace(alpha, fix, 1)
            corrected.append(tok)
        out.append(' '.join(corrected))

    return '\n'.join(out)


def light_correct(text: str, max_words: int = 150) -> str:
    """
    Lightweight TextBlob-based spell correction.
    Capped at `max_words` to avoid slowness on long documents.
    Falls back silently if textblob is not installed.
    """
    try:
        from textblob import TextBlob  # type: ignore
    except ImportError:
        return text

    words = text.split()
    if len(words) > max_words:
        # Correct only the first max_words portion, leave the rest as-is
        head = ' '.join(words[:max_words])
        tail = ' '.join(words[max_words:])
        try:
            corrected_head = str(TextBlob(head).correct())
        except Exception:
            corrected_head = head
        return corrected_head + ' ' + tail if tail else corrected_head

    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text


def spellcheck(text: str) -> str:
    """
    Combined spellcheck pipeline:
    - word_level_correct  (fast, preserves structure)
    - light_correct       (TextBlob, more accurate but capped)
    """
    text = word_level_correct(text)
    text = light_correct(text)
    return text


def postprocess(text: str, fix_structure: bool = True) -> str:
    """
    Full post-processing pipeline:
    1. Fix merged list-item markers
    2. Correct common OCR mistakes
    3. Normalise financial acronyms to uppercase
    4. Normalise whitespace
    5. Restore list / document structure
    """
    if not text:
        return text
    text = fix_merged_list_items(text)
    text = fix_numbered_list(text)
    text = correct_text(text)        # exact word-level fixes
    text = correct_phrases(text)     # multi-word phrase reconstruction
    text = fuzzy_domain_correct(text)  # fuzzy domain vocab matching
    text = normalize_acronyms(text)
    text = normalize_whitespace(text)
    if fix_structure:
        text = restore_structure(text)
    return text
