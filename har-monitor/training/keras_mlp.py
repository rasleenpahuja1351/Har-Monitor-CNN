"""Keras Dense (MLP) model for UCI HAR, with subject-wise validation."""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from har_load import load_har, subject_split

SEED = 42
keras.utils.set_random_seed(SEED)

data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
X_train, y_train, s_train, X_test, y_test, s_test, le = load_har(data_dir)
n_classes = len(le.classes_)

# Subject-wise split: validation subjects are never seen in training
tr, va = subject_split(X_train, y_train, s_train)

# Scale features (fit on training part only, then reuse everywhere)
scaler = StandardScaler().fit(X_train.iloc[tr])
X_tr = scaler.transform(X_train.iloc[tr]).astype("float32")
X_va = scaler.transform(X_train.iloc[va]).astype("float32")
X_te = scaler.transform(X_test).astype("float32")
y_tr, y_va = y_train[tr], y_train[va]


def build_model(n_features, n_classes):
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for units, drop in [(512, 0.4), (256, 0.4), (128, 0.3)]:
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(drop)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = build_model(X_tr.shape[1], n_classes)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                  restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_har_mlp.keras", monitor="val_loss",
                                    save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=5, min_lr=1e-5),
]

history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                    epochs=200, batch_size=64, callbacks=callbacks, verbose=2)

# Save curves for the dashboard (used by scripts/export_model.py --history)
import json
with open("history.json", "w") as f:
    json.dump({k: [float(v) for v in history.history[k]]
               for k in ["accuracy", "val_accuracy", "loss", "val_loss"]}, f)

# Load the best checkpoint and evaluate
best = keras.models.load_model("best_har_mlp.keras")
val_acc = accuracy_score(y_va, best.predict(X_va, verbose=0).argmax(1))
y_pred = best.predict(X_te, verbose=0).argmax(1)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nBest epoch: {int(np.argmin(history.history['val_loss'])) + 1}")
print(f"Keras validation accuracy (unseen subjects): {val_acc:.4f}")
print(f"Keras test accuracy: {test_acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

# Save scaler + label encoder for FastAPI
joblib.dump({"scaler": scaler, "label_encoder": le}, "har_mlp_preprocess.joblib")

# ---- Training curves ----
h = history.history
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
    ax.plot(h[key], label="train")
    ax.plot(h[f"val_{key}"], label="validation (unseen subjects)")
    ax.axvline(int(np.argmin(h["val_loss"])), color="gray", ls="--", lw=1, label="best epoch")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("keras_training_curves.png", dpi=150)

# ---- Confusion matrix ----
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=le.classes_).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title(f"Keras MLP - Test confusion matrix (acc {test_acc:.3f})")
plt.tight_layout()
plt.savefig("keras_confusion_matrix.png", dpi=150)

# ---- Comparison with Random Forest (same subject split) ----
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
rf.fit(X_train.iloc[tr], y_tr)
rf_val = accuracy_score(y_va, rf.predict(X_train.iloc[va]))
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n=== Comparison (test set) ===")
print(f"{'Model':<14}{'Val acc':>9}{'Test acc':>10}{'Macro F1':>10}")
print(f"{'RandomForest':<14}{rf_val:>9.4f}{accuracy_score(y_test, rf_pred):>10.4f}"
      f"{f1_score(y_test, rf_pred, average='macro'):>10.4f}")
print(f"{'Keras MLP':<14}{val_acc:>9.4f}{test_acc:>10.4f}"
      f"{f1_score(y_test, y_pred, average='macro'):>10.4f}")

print("\nPer-class F1 (test):")
rf_f1 = f1_score(y_test, rf_pred, average=None)
nn_f1 = f1_score(y_test, y_pred, average=None)
for name, a, b in zip(le.classes_, rf_f1, nn_f1):
    print(f"  {name:<20} RF {a:.3f}   Keras {b:.3f}")
