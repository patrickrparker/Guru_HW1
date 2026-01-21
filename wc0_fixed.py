#!/usr/bin/env python3 -B
"""Word frequency counter - refactored version (wc0_fixed.py)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# ----------------------------
# Q3 / AQ3: Policy extracted into CONFIG with no hard-coded "rules" in logic
# ----------------------------
CONFIG = {
    # Policy: which file to analyze (mechanism supports any path)
    "default_file": "essay.txt",

    # Policy: what counts as punctuation to strip
    "punctuation": '.,!?;:"()[]',

    # Policy: how many results to show
    "top_n": 10,

    # Policy: output format options (bonus)
    "output_format": "report",  # report | json | csv

    # Bonus policy: language selection (supports different stopwords)
    "language": "en",

    # Bonus policy: optional stopwords file (if present, overrides defaults)
    "stopwords_file": "stopwords.txt",

    # Policy data: stopwords by language (can be expanded)
    "stopwords_by_lang": {
        "en": [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "is", "was", "are", "were", "be", "been", "with",
        ],
        # Bonus: example Spanish stopwords list (small starter set)
        "es": ["el", "la", "los", "las", "y", "o", "pero", "en", "de", "es", "son"],
    },
}


# ----------------------------
# Q1 / AQ1: Separation of Concerns (SoC)
# - Model: pure data processing functions (NO printing, NO CLI parsing)
# - View: formatting functions that *return strings*
# - Controller: main() wires I/O + model + view
# ----------------------------

@dataclass(frozen=True)
class AnalysisResult:
    filename: str
    total_words: int
    unique_words: int
    top: List[Tuple[str, int]]  # (word, count)


# ============================
# Model (no printing here)
# ============================

def read_text(path: str) -> str:
    """Read a whole file into a string."""
    with open(path) as f:
        return f.read()


def iter_tokens_from_file(path: str) -> Iterable[str]:
    """Q1/AQ1: Stream tokens from a file (no whole-file read)."""
    with open(path) as f:
        for line in f:
            for tok in line.lower().split():
                yield tok


def analyze_file(filename: str, cfg: dict) -> AnalysisResult:
    """Q1/AQ1: Analyze a file by streaming it line-by-line."""
    stopwords = get_stopwords(cfg)
    tokens = iter_tokens_from_file(filename)
    words = iter_clean_words(tokens, cfg["punctuation"])
    counts = count_frequencies(words, stopwords)
    items = sort_by_frequency(counts)
    top = top_n_items(items, cfg["top_n"])
    return compute_result(filename, counts, top)


def normalize(text: str) -> List[str]:
    """Lowercase and split into raw tokens."""
    return text.lower().split()


def clean_word(word: str, punctuation: str) -> str:
    """Bonus test target: remove punctuation from ends of a token."""
    return word.strip(punctuation)


def iter_clean_words(tokens: Iterable[str], punctuation: str) -> Iterable[str]:
    """Yield cleaned tokens (empty tokens omitted)."""
    for t in tokens:
        w = clean_word(t, punctuation)
        if w:
            yield w


def load_stopwords_from_file(path: str) -> List[str]:
    """Bonus: load stopwords from a file (one per line; ignores blanks/#)."""
    words: List[str] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                words.append(s)
    return words


def get_stopwords(cfg: dict) -> List[str]:
    """Q3/AQ3: stopwords are policy; mechanism can load defaults or a file."""
    file_path = cfg.get("stopwords_file", "")
    if file_path and os.path.exists(file_path):
        return load_stopwords_from_file(file_path)
    lang = cfg.get("language", "en")
    return list(cfg["stopwords_by_lang"].get(lang, []))


def is_stopword(word: str, stopwords: Sequence[str]) -> bool:
    """Small, focused predicate."""
    return word in stopwords


def count_frequencies(words: Iterable[str], stopwords: Sequence[str]) -> Dict[str, int]:
    """Q2/AQ2: Single job: count words while skipping stopwords."""
    counts: Dict[str, int] = {}
    for w in words:
        if not is_stopword(w, stopwords):
            counts[w] = counts.get(w, 0) + 1
    return counts


def sort_by_frequency(counts: Dict[str, int]) -> List[Tuple[str, int]]:
    """Return (word,count) pairs sorted by count descending."""
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def top_n_items(items: Sequence[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    """Return the first n items."""
    return list(items[:n])


def compute_result(filename: str, counts: Dict[str, int], top: List[Tuple[str, int]]) -> AnalysisResult:
    """Small function: compute summary stats and package results."""
    return AnalysisResult(
        filename=filename,
        total_words=sum(counts.values()),
        unique_words=len(counts),
        top=top,
    )


def analyze_text(filename: str, text: str, cfg: dict) -> AnalysisResult:
    """Q2/AQ2: 'orchestrator' that composes small model functions."""
    stopwords = get_stopwords(cfg)
    tokens = normalize(text)
    words = iter_clean_words(tokens, cfg["punctuation"])
    counts = count_frequencies(words, stopwords)
    ranked = sort_by_frequency(counts)
    top = top_n_items(ranked, cfg["top_n"])
    return compute_result(filename, counts, top)


# ============================
# View (formatting only; still no printing)
# ============================

def header_block(filename: str) -> str:
    """Return the report header with the exact spacing from wc0.py."""
    line = "=" * 50
    return f"\n{line}\nWORD FREQUENCY ANALYSIS - {filename}\n{line}\n\n"


def format_stats(result: AnalysisResult) -> str:
    """Return the stats lines."""
    return (
        f"Total words (after removing stopwords): {result.total_words}\n"
        f"Unique words: {result.unique_words}\n\n"
    )


def format_top_header(n: int) -> str:
    """Return the 'Top N' heading."""
    return f"Top {n} most frequent words:\n\n"


def format_bar(count: int) -> str:
    """Q4/AQ4: Tiny function for a single formatting task."""
    return "*" * count


def format_top_rows(pairs: Sequence[Tuple[str, int]]) -> str:
    """Return the ranked rows, matching wc0.py's spacing."""
    lines: List[str] = []
    for i, (word, count) in enumerate(pairs, 1):
        bar = format_bar(count)
        lines.append(f"{i:2}. {word:15} {count:3} {bar}")
    return "\n".join(lines) + "\n\n"


def to_report(result: AnalysisResult, cfg: dict) -> str:
    """Default presentation: matches wc0.py output EXACTLY."""
    return (
        header_block(result.filename)
        + format_stats(result)
        + format_top_header(cfg["top_n"])
        + format_top_rows(result.top)
    )


# ----------------------------
# Bonus output formats (JSON/CSV)
# ----------------------------

def result_to_dict(result: AnalysisResult) -> dict:
    """Convert AnalysisResult to a plain dict."""
    return {
        "file": result.filename,
        "total_words": result.total_words,
        "unique_words": result.unique_words,
        "top": [{"word": w, "count": c} for (w, c) in result.top],
    }


def toJSON(result: AnalysisResult) -> str:
    """Bonus: JSON output."""
    return json.dumps(result_to_dict(result), indent=2, sort_keys=True)


def toCSV(result: AnalysisResult) -> str:
    """Bonus: CSV output."""
    rows = ["word,count"]
    rows += [f"{w},{c}" for (w, c) in result.top]
    return "\n".join(rows)


def render(result: AnalysisResult, cfg: dict) -> str:
    """Controller helper: choose a view based on output policy."""
    fmt = cfg.get("output_format", "report").lower()
    if fmt == "json":
        return toJSON(result) + "\n"
    if fmt == "csv":
        return toCSV(result) + "\n"
    return to_report(result, cfg)


# ============================
# Controller (I/O, CLI)
# ============================

def parse_args(argv: Sequence[str], cfg: dict) -> dict:
    """Q1/AQ1: CLI parsing lives here (not inside the model)."""
    out = dict(cfg)
    # Accept an optional positional filename argument anywhere after the script name.
    for a in argv[1:]:
        if not a.startswith('-'):
            out["default_file"] = a
            break
    if "--json" in argv:
        out["output_format"] = "json"
    if "--csv" in argv:
        out["output_format"] = "csv"
    return out
def test_cleanWord() -> None:
    assert clean_word("hello,", ".,") == "hello"


def test_countWords() -> None:
    counts = count_frequencies(["the", "cat", "and", "the", "dog"], ["the" ])
    assert counts == {"cat": 1, "and": 1, "dog": 1}


def run_tests() -> None:
    test_cleanWord()
    test_countWords()


def main(argv: Sequence[str]) -> int:
    """AQ1: Controller orchestrates I/O -> model -> view -> printing."""
    if "--test" in argv:
        run_tests()
        return 0

    cfg = parse_args(list(argv), CONFIG)
    filename = cfg["default_file"]
    result = analyze_file(filename, cfg)
    sys.stdout.write(render(result, cfg))  # printing ONLY happens here
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
