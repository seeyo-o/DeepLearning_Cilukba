"""Pemeriksaan masukan (fitur unik 2) dan prediksi satu teks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from stylometry import clean_text, count_words

DetectorFactory.seed = 0  # hasil deteksi bahasa deterministik (NF3)


@dataclass
class InputCheck:
    status: str            # "ok", "kosong", atau "bukan_inggris"
    n_words: int
    warnings: List[str] = field(default_factory=list)
    message: str = ""


def detect_english(text: str) -> bool:
    """True bila bahasa dengan peluang terbesar menurut langdetect adalah Inggris."""
    try:
        return detect_langs(text)[0].lang == "en"
    except LangDetectException:
        return False


def check_input(text: str | None, q1: int, l_max: int) -> InputCheck:
    """Validasi teks: kosong ditolak, non-Inggris ditolak, panjang di luar
    kisaran (N <= Q1 atau N > L_maks) tetap diproses dengan peringatan."""
    text = clean_text(text)
    if text == "":
        return InputCheck("kosong", 0, message="Teks ulasan wajib diisi.")
    n = count_words(text)
    if n == 0 or not detect_english(text):
        return InputCheck("bukan_inggris", n,
                          message="Hanya teks ulasan berbahasa Inggris yang didukung.")
    warnings = []
    if n <= q1:
        warnings.append(f"Teks terlalu pendek ({n} kata, panjang yang disarankan "
                        f"{q1 + 1} sampai {l_max} kata). Hasil deteksi pada teks yang "
                        "sangat pendek kurang dapat diandalkan.")
    if n > l_max:
        warnings.append(f"Teks terlalu panjang ({n} kata, melebihi {l_max} kata). "
                        "Teks lebih panjang daripada ulasan AI yang dipakai saat pelatihan, "
                        "sehingga hasil deteksi kurang dapat diandalkan.")
    return InputCheck("ok", n, warnings)
