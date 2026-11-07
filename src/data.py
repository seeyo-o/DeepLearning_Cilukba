"""Pemuatan dataset Modern Fake Reviews, pembersihan, dan penelusuran src_idx."""
from __future__ import annotations

import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from stylometry import FEATURE_NAMES, clean_text, count_words, extract_features

SPLIT_SEED: int = 1998
HF_REPOS: Dict[str, str] = {
    "deepseek": "Flowerly/modern-fake-reviews",
    "glm": "Flowerly/modern-fake-reviews-glm",
    "lfm": "Flowerly/modern-fake-reviews-lfm",
}
GITHUB_RAW: str = "https://raw.githubusercontent.com/PurplexyFlower/Fake-review/main/dataset/"
GITHUB_FILES: Dict[str, str] = {
    "deepseek": "modern_reviews_deepseek.csv",
    "glm": "modern_reviews_glm.csv",
    "lfm": "modern_reviews_lfm.csv",
    "raw_deepseek": "sota_fakes_deepseek.csv",
    "original": "fake reviews dataset.csv",
}
LABEL_MAP: Dict[str, int] = {"OR": 0, "CG": 1}


def _download(name: str, data_dir: Path) -> Path:
    """Unduh berkas CSV pembuat dataset dari GitHub bila belum tersedia."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / GITHUB_FILES[name]
    if not path.exists():
        url = GITHUB_RAW + urllib.request.quote(GITHUB_FILES[name])
        urllib.request.urlretrieve(url, path)
    return path


def load_official_splits(variant: str, data_dir: Path) -> Tuple[pd.DataFrame, ...]:
    """Muat pembagian resmi (train/validation/test) untuk satu varian dataset.

    Urutan sumber: (1) Hugging Face Hub; (2) bila Hub tidak dapat diakses,
    CSV resmi di GitHub pembuat dataset dibagi ulang dengan prosedur yang
    didokumentasikan pembuatnya: train_test_split(test_size=0.2, seed=1998),
    lalu sisa 20% dibagi dengan test_size=0.7, seed=1998.
    """
    from datasets import Dataset, load_dataset

    try:
        hub = load_dataset(HF_REPOS[variant])
        frames = [hub[s].to_pandas() for s in ("train", "validation", "test")]
        source = "Hugging Face Hub"
    except Exception:  # noqa: BLE001 - jaringan dapat diblokir
        full = pd.read_csv(_download(variant, data_dir))
        first = Dataset.from_pandas(full, preserve_index=False).train_test_split(
            test_size=0.2, seed=SPLIT_SEED)
        second = first["test"].train_test_split(test_size=0.7, seed=SPLIT_SEED)
        frames = [first["train"].to_pandas(), second["train"].to_pandas(),
                  second["test"].to_pandas()]
        source = "GitHub (split ulang seed 1998)"
    for frame in frames:
        frame["y"] = frame["label"].map(LABEL_MAP).astype(int)
    print(f"[{variant}] sumber data: {source}; ukuran: {[len(f) for f in frames]}")
    return tuple(frames)


def clean_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pembersihan minimal tanpa mengubah huruf kapital dan tanda baca.

    Langkah: normalisasi spasi, keluarkan teks kosong, keluarkan duplikat persis
    di data training, keluarkan teks training yang sama dengan validation/test.
    Data validation dan testing resmi tidak dikurangi.
    """
    report = {}
    frames = {"train": train.copy(), "val": val.copy(), "test": test.copy()}
    for name, frame in frames.items():
        frame["text_"] = frame["text_"].map(clean_text)
        report[(name, "empty")] = int((frame["text_"] == "").sum())
        report[(name, "dup_within")] = int(frame["text_"].duplicated().sum())
        report[(name, "conflict_label")] = int(
            frame.groupby("text_")["y"].nunique().gt(1).sum())
        frames[name] = frame[frame["text_"] != ""]
    tr = frames["train"]
    tr = tr[~tr["text_"].duplicated()]
    held = set(frames["val"]["text_"]) | set(frames["test"]["text_"])
    report[("train", "overlap_val_test")] = int(tr["text_"].isin(held).sum())
    tr = tr[~tr["text_"].isin(held)]
    report[("val", "overlap_test")] = int(
        frames["val"]["text_"].isin(set(frames["test"]["text_"])).sum())
    frames["train"] = tr
    rep = pd.Series(report).unstack()
    return frames["train"].reset_index(drop=True), frames["val"].reset_index(
        drop=True), frames["test"].reset_index(drop=True), rep


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan 11 kolom fitur stilometrik dan jumlah kata."""
    feats = pd.DataFrame([extract_features(t) for t in frame["text_"]])
    out = pd.concat([frame.reset_index(drop=True), feats], axis=1)
    out["n_words"] = out["text_"].map(count_words)
    out["rating"] = out["rating"].astype(float)
    return out


# ---- Penelusuran ulasan manusia sumber (src_idx) ---------------------------
_SMART = {0x2018: "'", 0x2019: "'", 0x201a: "'", 0x201b: "'", 0x201c: '"',
          0x201d: '"', 0x201e: '"', 0x2032: "'", 0x2033: '"', 0x2013: ", ",
          0x2014: ", ", 0x2026: "...", 0x00a0: " "}


def creator_normalize(text: str) -> str:
    """Replika fungsi normalize_text milik pembuat dataset (build_modern_dataset.py)."""
    text = str(text).translate(_SMART)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*,", ", ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r",(\s*[.!?])", r"\1", text)
    return text.strip().strip(",").strip()


def trace_sources(test: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
                  data_dir: Path) -> pd.DataFrame:
    """Temukan lokasi ulasan manusia sumber untuk setiap ulasan AI data testing."""
    raw = pd.read_csv(_download("raw_deepseek", data_dir))
    orig = pd.read_csv(_download("original", data_dir))
    raw["key"] = raw["text_"].map(creator_normalize).map(clean_text)
    gen_to_src = dict(zip(raw["key"], raw["src_idx"]))
    location = {}
    for name, frame in (("test", test), ("val", val), ("train", train)):
        for text in frame.loc[frame["y"] == 0, "text_"]:
            location[text] = name
    rows = []
    for _, row in test[test["y"] == 1].iterrows():
        src = gen_to_src.get(row["text_"])
        if src is None or pd.isna(src):
            rows.append({"text_": row["text_"], "src_idx": np.nan, "src_loc": "tidak ditemukan",
                         "same_cat_rating": np.nan})
            continue
        src_row = orig.iloc[int(src)]
        src_text = clean_text(creator_normalize(src_row["text_"]))
        rows.append({
            "text_": row["text_"], "src_idx": int(src),
            "src_loc": location.get(src_text, "tidak ada di dataset akhir"),
            "same_cat_rating": bool(src_row["category"] == row["category"]
                                    and float(src_row["rating"]) == float(row["rating"])),
        })
    return pd.DataFrame(rows)
