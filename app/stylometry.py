"""Ekstraksi 11 fitur stilometrik untuk deteksi ulasan produk hasil generatif AI.

Definisi fitur mengikuti subbab 2.9 skripsi kelompok CILUKBA:
kata = deretan huruf yang boleh memuat satu apostrof di tengahnya,
kalimat = potongan teks yang dipisahkan tanda titik, seru, tanya, atau elipsis
(deretan tanda berurutan dihitung satu pemisah).
Modul ini dipakai SAMA PERSIS saat pelatihan dan saat inferensi (aplikasi web).
"""
from __future__ import annotations

import re
from typing import Dict, List

FEATURE_NAMES: List[str] = [
    "mattr", "avg_word_len", "function_word_ratio", "avg_sentence_len",
    "comma_ratio", "exclamation_ratio", "apostrophe_ratio",
    "informal_punct_ratio", "uppercase_ratio", "digit_ratio",
    "lowercase_start_ratio",
]
MATTR_WINDOW: int = 25
APOSTROPHES: tuple = ("'", "’")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
SENT_SPLIT_RE = re.compile(r"(?:[.!?]|…)+")
INFORMAL_RE = re.compile(r"\.{3,}|…|!{2,}|\?{2,}")

FUNCTION_WORDS = set([
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'each', 'every', 'either',
    'neither', 'some', 'any', 'no', 'all', 'both', 'few', 'many', 'much', 'more',
    'most', 'several', 'such', 'another', 'other', 'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours',
    'ourselves', 'they', 'them', 'their', 'theirs', 'themselves', 'who', 'whom',
    'whose', 'which', 'what', 'someone', 'something', 'anyone', 'anything', 'everyone',
    'everything', 'nobody', 'nothing', "i'm", "i've", "i'll", "i'd", "you're",
    "you've", "you'll", "you'd", "he's", "he'll", "he'd", "she's", "she'll", "she'd",
    "it's", "it'll", "we're", "we've", "we'll", "we'd", "they're", "they've",
    "they'll", "they'd", "that's", "there's", "what's", "who's", "let's", "isn't",
    "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "haven't", "hasn't",
    "hadn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "mustn't",
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as',
    'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by',
    'despite', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'like', 'near',
    'of', 'off', 'on', 'onto', 'out', 'outside', 'over', 'past', 'since', 'through',
    'throughout', 'till', 'to', 'toward', 'towards', 'under', 'until', 'up', 'upon',
    'with', 'within', 'without', 'and', 'but', 'or', 'nor', 'so', 'yet', 'because',
    'although', 'though', 'while', 'whereas', 'if', 'unless', 'whether', 'than',
    'once', 'when', 'whenever', 'where', 'wherever', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'can',
    'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',])


def clean_text(text: str | None) -> str:
    """Normalisasi spasi: hapus spasi di tepi dan ringkas spasi berulang."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_words(text: str) -> List[str]:
    """Kembalikan daftar kata (huruf, boleh satu apostrof di tengah)."""
    return WORD_RE.findall(text)


def count_words(text: str) -> int:
    """Hitung jumlah kata dengan definisi yang sama dengan fitur."""
    return len(tokenize_words(text))


def split_sentences(text: str) -> List[str]:
    """Pecah teks menjadi kalimat; potongan tanpa huruf/angka diabaikan."""
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if re.search(r"[^\W_]", p)]


def _normalize_apostrophe(word: str) -> str:
    return word.replace("’", "'")


def mattr(words_lower: List[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio; memakai TTR biasa jika N < window."""
    n = len(words_lower)
    if n == 0:
        return 0.0
    if n < window:
        return len(set(words_lower)) / n
    ttr_values = [
        len(set(words_lower[i:i + window])) / window
        for i in range(n - window + 1)
    ]
    return sum(ttr_values) / len(ttr_values)


def extract_features(text: str | None) -> Dict[str, float]:
    """Hitung 11 fitur stilometrik untuk satu teks ulasan."""
    text = clean_text(text)
    words = tokenize_words(text)
    n_words = len(words)
    if n_words == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    words_lower = [_normalize_apostrophe(w).lower() for w in words]
    sentences = split_sentences(text)
    n_sent = max(len(sentences), 1)

    letters_in_words = sum(sum(ch.isalpha() for ch in w) for w in words)
    letters_all = [ch for ch in text if ch.isalpha()]
    n_letters = len(letters_all)
    n_upper = sum(ch.isupper() for ch in letters_all)
    non_space = [ch for ch in text if not ch.isspace()]
    n_non_space = len(non_space)
    n_digits = sum(ch.isdigit() for ch in non_space)

    lower_start = 0
    for sentence in sentences:
        first = next((ch for ch in sentence if ch.isalpha()), None)
        if first is not None and first.islower():
            lower_start += 1

    return {
        "mattr": mattr(words_lower),
        "avg_word_len": letters_in_words / n_words,
        "function_word_ratio": sum(w in FUNCTION_WORDS for w in words_lower) / n_words,
        "avg_sentence_len": n_words / n_sent,
        "comma_ratio": text.count(",") / n_words,
        "exclamation_ratio": text.count("!") / n_words,
        "apostrophe_ratio": sum(text.count(a) for a in APOSTROPHES) / n_words,
        "informal_punct_ratio": len(INFORMAL_RE.findall(text)) / n_words,
        "uppercase_ratio": n_upper / n_letters if n_letters else 0.0,
        "digit_ratio": n_digits / n_non_space if n_non_space else 0.0,
        "lowercase_start_ratio": lower_start / n_sent,
    }
