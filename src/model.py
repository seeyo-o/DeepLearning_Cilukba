"""Arsitektur MLP (DNN) dan prosedur pelatihan dengan TensorFlow/Keras."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

DROPOUT: float = 0.1
BATCH_SIZE: int = 32
MAX_EPOCHS: int = 20
PATIENCE: int = 3
WEIGHT_DECAY: float = 0.01


def set_global_seed(seed: int) -> None:
    """Kunci seluruh sumber keacakan (Python, NumPy, TensorFlow)."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_model(arch: str, x_train: np.ndarray, regularized: bool = True) -> tf.keras.Model:
    """Bangun MLP M1 (Dense 256) atau M2 (Dense 64 -> Dense 32).

    Layer Normalization di-adapt HANYA pada data training (z-score) dan ikut
    tersimpan di dalam model sehingga inferensi memakai statistik yang sama.
    """
    norm = tf.keras.layers.Normalization(axis=-1, name="zscore")
    norm.adapt(x_train.astype("float32"))
    inputs = tf.keras.Input(shape=(x_train.shape[1],), name="fitur")
    x = norm(inputs)
    if arch == "M1":
        x = tf.keras.layers.Dense(256, activation="relu", name="hidden_1")(x)
        if regularized:
            x = tf.keras.layers.Dropout(DROPOUT, name="dropout_1")(x)
    elif arch == "M2":
        for i, units in enumerate((64, 32), start=1):
            x = tf.keras.layers.Dense(units, name=f"hidden_{i}")(x)
            if regularized:
                x = tf.keras.layers.BatchNormalization(name=f"batchnorm_{i}")(x)
            x = tf.keras.layers.Activation("relu", name=f"relu_{i}")(x)
            if regularized:
                x = tf.keras.layers.Dropout(DROPOUT, name=f"dropout_{i}")(x)
    else:
        raise ValueError(f"Arsitektur tidak dikenal: {arch}")
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs, outputs, name=f"MLP_{arch}")


def make_dataset(x: np.ndarray, y: np.ndarray, seed: int, shuffle: bool) -> tf.data.Dataset:
    """Pipeline tf.data: from_tensor_slices -> shuffle -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y.astype("int32")))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class AurocCheckpoint(tf.keras.callbacks.Callback):
    """Hitung AUROC data validation tiap epoch, simpan bobot terbaik, early stopping."""

    def __init__(self, x_val: np.ndarray, y_val: np.ndarray, patience: Optional[int]):
        super().__init__()
        self.x_val, self.y_val, self.patience = x_val, y_val, patience
        self.best, self.best_epoch, self.wait, self.best_weights = -np.inf, 0, 0, None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        p = self.model.predict(self.x_val, verbose=0)[:, 1]
        auroc = float(roc_auc_score(self.y_val, p))
        logs = logs if logs is not None else {}
        logs["val_auroc"] = auroc
        if auroc > self.best:
            self.best, self.best_epoch, self.wait = auroc, epoch + 1, 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.patience is not None and self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        if self.patience is not None and self.best_weights is not None:
            self.model.set_weights(self.best_weights)


@dataclass
class TrainResult:
    model: tf.keras.Model
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_auroc: float
    epochs_run: int = field(default=0)


def train_model(arch: str, x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray,
                y_va: np.ndarray, seed: int, lr: float = 1e-3,
                regularized: bool = True) -> TrainResult:
    """Latih satu model. regularized=False = skenario S3 (tanpa dropout/BN,
    SGD learning rate tetap 0,01, tanpa early stopping, 20 epoch)."""
    set_global_seed(seed)
    model = build_model(arch, x_tr, regularized=regularized)
    if regularized:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
        patience: Optional[int] = PATIENCE
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        patience = None
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    callback = AurocCheckpoint(x_va, y_va, patience)
    hist = model.fit(make_dataset(x_tr, y_tr, seed, True), epochs=MAX_EPOCHS,
                     validation_data=make_dataset(x_va, y_va, seed, False),
                     callbacks=[callback], verbose=0)
    return TrainResult(model, hist.history, callback.best_epoch, callback.best,
                       len(hist.history["loss"]))
