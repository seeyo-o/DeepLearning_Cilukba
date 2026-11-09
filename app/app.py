"""Aplikasi web Gradio: deteksi ulasan produk hasil generatif AI (MLP + fitur stilometrik).

Model yang dipasang adalah model S1 seed 1998 yang dievaluasi di laporan,
dikompresi ke TensorFlow Lite (float16). Ekstraksi fitur memakai modul
stylometry.py yang sama persis dengan saat pelatihan.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np

from inference import check_input
from stylometry import clean_text, count_words, extract_features

try:  # LiteRT (pengganti tflite_runtime); TensorFlow tidak diperlukan
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover - cadangan bila LiteRT tidak tersedia
    from tensorflow.lite import Interpreter  # type: ignore

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INTERPRETER = Interpreter(model_path=str(HERE / "mlp_s1.tflite"))
INTERPRETER.allocate_tensors()
_IN = INTERPRETER.get_input_details()[0]["index"]
_OUT = INTERPRETER.get_output_details()[0]["index"]
Q1, L_MAX, T_STAR = CONFIG["q1"], CONFIG["l_max"], CONFIG["t_star"]
DISCLAIMER = ("Hasil ini hanya dugaan asal penulisan, bukan kepastian dan bukan "
              "penilaian atas niat penulis.")
EXAMPLES = [
    ["I bought this for my daughter and she loves it. The colors are bright, the material "
     "feels sturdy, and it survived a week of daily use without any scratches. Shipping "
     "was quick too."],
    ["got it yesterday... honestly not sure yet!! the strap broke after 2 days but the "
     "seller replaced it fast so i guess its ok"],
    ["Produk ini sangat bagus dan pengirimannya cepat sekali."],
]


def predict_proba(text: str) -> float:
    """Peluang ulasan ditulis AI menurut model TFLite."""
    feats = extract_features(text)
    x = np.array([[feats[name] for name in CONFIG["features"]]], dtype=np.float32)
    INTERPRETER.set_tensor(_IN, x)
    INTERPRETER.invoke()
    return float(INTERPRETER.get_tensor(_OUT)[0, 1])


def reliability_group(n_words: int) -> dict:
    """Fitur unik 3: pilih FPR/FNR data testing sesuai kelompok panjang teks."""
    for group in CONFIG["length_groups"]:
        if group["min"] <= n_words <= group["max"]:
            return group
    return CONFIG["length_groups"][-1]


def word_info(text: str) -> str:
    return (f"Jumlah kata: **{count_words(clean_text(text))}** · Panjang yang disarankan: "
            f"**{Q1 + 1} sampai {L_MAX} kata**")


def analyze(text: str) -> Tuple[str, str, str]:
    """Jalankan pemeriksaan masukan, prediksi, dan susun laporan hasil."""
    check = check_input(text, Q1, L_MAX)
    if check.status != "ok":
        return f"⚠️ {check.message}", "", word_info(text or "")
    p = predict_proba(clean_text(text))
    category = "Terindikasi ditulis AI" if p >= T_STAR else "Tidak terindikasi ditulis AI"
    group = reliability_group(check.n_words)
    warn = "".join(f"> ⚠️ {w}\n\n" for w in check.warnings)
    fpr = f"{group['fpr'] * 100:.2f}%".replace(".", ",") if group["fpr"] is not None else "-"
    fnr = f"{group['fnr'] * 100:.2f}%".replace(".", ",") if group["fnr"] is not None else "-"
    p_txt, t_txt = f"{p:.4f}".replace(".", ","), f"{T_STAR:.4f}".replace(".", ",")
    report = (f"{warn}### Kategori dugaan: {category}\n\n"
              f"**Keandalan model pada data uji ({group['name']}):**\n\n"
              f"| Ulasan manusia keliru dideteksi sebagai AI (FPR) | Ulasan AI gagal terdeteksi (FNR) |\n"
              f"|:---:|:---:|\n| {fpr} | {fnr} |\n\n"
              f"Peluang ditulis AI menurut model: {p_txt} (threshold t* = {t_txt})\n\n"
              f"> {DISCLAIMER}")
    return "", report, word_info(text)


def clear() -> Tuple[str, str, str, str]:
    return "", "", "", word_info("")


with gr.Blocks(title="Deteksi Ulasan Produk Buatan AI") as demo:
    gr.Markdown("# 🔎 Deteksi Ulasan Produk Buatan AI\n🔒 *Teks dan hasil tidak disimpan*")
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="Teks Ulasan", lines=10,
                                 placeholder="Tempel teks ulasan berbahasa Inggris di sini")
            info = gr.Markdown(word_info(""))
            message = gr.Markdown()
            with gr.Row():
                btn_clear = gr.Button("Bersihkan")
                btn_check = gr.Button("Periksa Ulasan", variant="primary")
            gr.Examples(EXAMPLES, inputs=text_in, label="Contoh masukan siap coba")
        with gr.Column():
            gr.Markdown("## Laporan Hasil Deteksi")
            report_out = gr.Markdown("*Hasil deteksi akan tampil di sini setelah tombol "
                                     "Periksa Ulasan ditekan.*")
    text_in.change(word_info, text_in, info)
    btn_check.click(analyze, text_in, [message, report_out, info])
    btn_clear.click(clear, None, [text_in, message, report_out, info])

if __name__ == "__main__":
    # Di Render, variabel PORT disediakan platform sehingga server mendengarkan di
    # 0.0.0.0:PORT. Saat dijalankan lokal tanpa PORT, buka http://127.0.0.1:7860.
    on_server = "PORT" in os.environ
    demo.launch(server_name="0.0.0.0" if on_server else "127.0.0.1",
                server_port=int(os.environ.get("PORT", 7860)))
