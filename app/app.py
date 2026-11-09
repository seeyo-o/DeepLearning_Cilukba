"""Aplikasi web Streamlit: deteksi ulasan produk hasil generatif AI (MLP + fitur stilometrik).

Model yang dipasang adalah model S1 seed 1998 yang dievaluasi di laporan,
dikompresi ke TensorFlow Lite (float16). Ekstraksi fitur memakai modul
stylometry.py yang sama persis dengan saat pelatihan.

Menjalankan secara lokal : streamlit run app.py
Deployment               : Streamlit Community Cloud, main file path app/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import streamlit as st

from inference import check_input
from stylometry import clean_text, count_words, extract_features

try:  # LiteRT (pengganti tflite_runtime); TensorFlow tidak diperlukan
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover - cadangan bila LiteRT tidak tersedia
    from tensorflow.lite import Interpreter  # type: ignore

HERE = Path(__file__).resolve().parent
DISCLAIMER = ("Hasil ini hanya dugaan asal penulisan, bukan kepastian dan bukan "
              "penilaian atas niat penulis.")
EXAMPLES: List[str] = [
    "I bought this for my daughter and she loves it. The colors are bright, the material "
    "feels sturdy, and it survived a week of daily use without any scratches. Shipping "
    "was quick too.",
    "got it yesterday... honestly not sure yet!! the strap broke after 2 days but the "
    "seller replaced it fast so i guess its ok",
    "Produk ini sangat bagus dan pengirimannya cepat sekali.",
]


@st.cache_resource
def load_model():
    """Muat config.json dan interpreter TFLite sekali saja untuk seluruh sesi."""
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    interpreter = Interpreter(model_path=str(HERE / "mlp_s1.tflite"))
    interpreter.allocate_tensors()
    return config, interpreter


CONFIG, INTERPRETER = load_model()
_IN = INTERPRETER.get_input_details()[0]["index"]
_OUT = INTERPRETER.get_output_details()[0]["index"]
Q1, L_MAX, T_STAR = CONFIG["q1"], CONFIG["l_max"], CONFIG["t_star"]


def predict_proba(text: str) -> float:
    """Peluang ulasan ditulis AI menurut model TFLite."""
    feats = extract_features(text)
    x = np.array([[feats[name] for name in CONFIG["features"]]], dtype=np.float32)
    INTERPRETER.set_tensor(_IN, x)
    INTERPRETER.invoke()
    return float(INTERPRETER.get_tensor(_OUT)[0, 1])


def reliability_group(n_words: int) -> Dict:
    """Fitur unik 3: pilih FPR/FNR data testing sesuai kelompok panjang teks."""
    for group in CONFIG["length_groups"]:
        if group["min"] <= n_words <= group["max"]:
            return group
    return CONFIG["length_groups"][-1]


def pct(value) -> str:
    return "-" if value is None else f"{value * 100:.2f}%".replace(".", ",")


def word_info(text: str) -> str:
    return (f"Jumlah kata: **{count_words(clean_text(text))}** · Panjang yang disarankan: "
            f"**{Q1 + 1} sampai {L_MAX} kata**")


def analyze(text: str) -> Dict:
    """Jalankan pemeriksaan masukan (fitur unik 2), prediksi, dan susun hasil."""
    check = check_input(text, Q1, L_MAX)
    if check.status != "ok":
        return {"status": "invalid", "message": check.message}
    p = predict_proba(clean_text(text))
    group = reliability_group(check.n_words)
    return {"status": "ok", "p": p, "is_ai": p >= T_STAR, "group": group,
            "warnings": check.warnings}


# ----------------------------- state & callback -----------------------------
if "text" not in st.session_state:
    st.session_state.text = ""
    st.session_state.result = None


def on_check() -> None:
    st.session_state.result = analyze(st.session_state.text)


def on_clear() -> None:
    st.session_state.text = ""
    st.session_state.result = None


def on_example(idx: int) -> None:
    st.session_state.text = EXAMPLES[idx]


# ---------------------------------- layout ----------------------------------
st.set_page_config(page_title="Deteksi Ulasan Produk Buatan AI", page_icon="🔎",
                   layout="wide")
st.title("🔎 Deteksi Ulasan Produk Buatan AI")
st.caption("🔒 Teks dan hasil tidak disimpan")

left, right = st.columns(2, gap="large")
with left:
    st.text_area("Teks Ulasan", key="text", height=230,
                 placeholder="Tempel teks ulasan berbahasa Inggris di sini")
    st.markdown(word_info(st.session_state.text))
    b1, b2 = st.columns(2)
    b1.button("Bersihkan", on_click=on_clear, width="stretch")
    b2.button("Periksa Ulasan", type="primary", on_click=on_check,
              width="stretch")
    st.markdown("**Contoh masukan siap coba**")
    for i, example in enumerate(EXAMPLES):
        label = example if len(example) <= 60 else example[:57] + "..."
        st.button(label, key=f"ex{i}", on_click=on_example, args=(i,))

with right:
    st.subheader("Laporan Hasil Deteksi")
    res = st.session_state.result
    if res is None:
        st.markdown("*Hasil deteksi akan tampil di sini setelah tombol "
                    "Periksa Ulasan ditekan.*")
    elif res["status"] == "invalid":
        st.warning(res["message"], icon="⚠️")
    else:
        for w in res["warnings"]:
            st.warning(w, icon="⚠️")
        category = ("Terindikasi ditulis AI" if res["is_ai"]
                    else "Tidak terindikasi ditulis AI")
        st.markdown(f"#### Kategori dugaan: {category}")
        group = res["group"]
        st.markdown(f"**Keandalan model pada data uji ({group['name']}):**")
        st.markdown(
            "| Ulasan manusia keliru dideteksi sebagai AI (FPR) | "
            "Ulasan AI gagal terdeteksi (FNR) |\n|:---:|:---:|\n"
            f"| {pct(group['fpr'])} | {pct(group['fnr'])} |")
        p_txt = f"{res['p']:.4f}".replace(".", ",")
        t_txt = f"{T_STAR:.4f}".replace(".", ",")
        st.markdown(f"Peluang ditulis AI menurut model: {p_txt} (threshold t* = {t_txt})")
        st.info(DISCLAIMER)
