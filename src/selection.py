"""Pemilihan fitur stilometrik berbasis data training saja (subbab 2.10 skripsi)."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.metrics import roc_auc_score

CORR_LIMIT: float = 0.9          # |rho| > 0,9 dianggap mengukur hal yang sama
LENGTH_CORR_LIMIT: float = 0.9   # |rho| > 0,9 terhadap jumlah kata = proksi panjang
ARTIFACT_FEATURES: List[str] = []  # tanda pisah sudah tidak dijadikan fitur


def select_features(train: pd.DataFrame, candidates: List[str]) -> Dict[str, object]:
    """Terapkan empat kriteria pemilihan fitur dan kembalikan tabel keputusannya."""
    auroc = {f: roc_auc_score(train["y"], train[f]) for f in candidates}
    power = {f: abs(a - 0.5) for f, a in auroc.items()}
    rho = train[candidates + ["n_words"]].corr(method="spearman")
    decision = {f: "Dipertahankan" for f in candidates}
    for i, fa in enumerate(candidates):
        for fb in candidates[i + 1:]:
            if abs(rho.loc[fa, fb]) > CORR_LIMIT:
                weaker = fa if power[fa] < power[fb] else fb
                decision[weaker] = f"Dikeluarkan: |rho| > {CORR_LIMIT} dengan fitur lebih kuat"
    for f in candidates:
        if abs(rho.loc[f, "n_words"]) > LENGTH_CORR_LIMIT:
            decision[f] = "Dikeluarkan: bergantung pada panjang ulasan"
        if f in ARTIFACT_FEATURES:
            decision[f] = "Dikeluarkan: artefak pembuatan data"
    table = pd.DataFrame({
        "AUROC": pd.Series(auroc), "Daya pembeda": pd.Series(power),
        "rho_jumlah_kata": rho.loc[candidates, "n_words"], "Keputusan": pd.Series(decision),
    }).sort_values("Daya pembeda", ascending=False)
    selected = [f for f in candidates if decision[f] == "Dipertahankan"]
    pairs = [(fa, fb, abs(rho.loc[fa, fb])) for i, fa in enumerate(candidates)
             for fb in candidates[i + 1:]]
    max_pair = sorted(pairs, key=lambda item: item[2], reverse=True)[:3]
    return {"table": table, "selected": selected, "spearman": rho,
            "max_pair": max_pair}
