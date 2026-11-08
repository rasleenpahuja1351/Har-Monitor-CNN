"""
Export the trained Keras model to plain NumPy weights for the API.

Why: TensorFlow is far too large for a Vercel function. The model is only
Dense + BatchNorm + ReLU layers, so we fold BatchNorm (and the StandardScaler)
into the Dense weights and run inference with NumPy (~2 MB of weights).

Run once, locally, after training (needs the training requirements):
    python scripts/export_model.py --data training/data --artifacts training
Writes:
    model/har_mlp.npz    weights for the API
    model/metrics.json   dashboard numbers (accuracy, F1, confusion matrix, curves)
    model/samples.npz    a stratified set of real test windows for the live demo
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from har_load import load_har, subject_split  # noqa: E402


def fold_model(model, scaler):
    """Return a list of (W, b) for each layer with BN and scaler folded in."""
    from tensorflow import keras

    layers_out = []
    dense, pending = None, None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w = layer.kernel.numpy().astype(np.float64)
            b = layer.bias.numpy().astype(np.float64) if layer.use_bias else np.zeros(w.shape[1])
            pending = [w, b]
            layers_out.append(pending)
        elif isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta = layer.gamma.numpy(), layer.beta.numpy()
            mean, var = layer.moving_mean.numpy(), layer.moving_variance.numpy()
            k = gamma / np.sqrt(var + layer.epsilon)
            pending[0] = pending[0] * k
            pending[1] = (pending[1] - mean) * k + beta

    # Fold StandardScaler into the first layer: (x - mu)/s @ W = x @ (W/s) - (mu/s) @ W
    w0, b0 = layers_out[0]
    mu, s = scaler.mean_, scaler.scale_
    layers_out[0] = [w0 / s[:, None], b0 - (mu / s) @ w0]
    return [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers_out]


def numpy_predict(layers, x):
    h = x.astype(np.float32)
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    h = h - h.max(axis=1, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.csv and test.csv")
    ap.add_argument("--artifacts", required=True,
                    help="folder with best_har_mlp.keras, har_mlp_preprocess.joblib, rf_baseline.joblib")
    ap.add_argument("--history", default=None, help="optional history.json from training")
    args = ap.parse_args()

    from tensorflow import keras

    art = Path(args.artifacts)
    model = keras.models.load_model(art / "best_har_mlp.keras")
    pre = joblib.load(art / "har_mlp_preprocess.joblib")
    scaler, le = pre["scaler"], pre["label_encoder"]

    X, y, s, Xt, yt, st, _ = load_har(args.data)
    feature_names = list(X.columns)

    layers = fold_model(model, scaler)

    # ---- parity check against Keras ----
    keras_p = model.predict(scaler.transform(Xt).astype("float32"), verbose=0)
    np_p = numpy_predict(layers, Xt.to_numpy())
    assert (keras_p.argmax(1) == np_p.argmax(1)).mean() > 0.999, "NumPy export does not match Keras"
    print("Max probability difference vs Keras:", float(np.abs(keras_p - np_p).max()))

    out = ROOT / "model"
    out.mkdir(exist_ok=True)
    arrays = {}
    for i, (w, b) in enumerate(layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = w, b
    np.savez_compressed(out / "har_mlp.npz", n_layers=len(layers),
                        classes=np.array([str(c) for c in le.classes_], dtype=str),
                        features=np.array(feature_names, dtype=str), **arrays)

    # ---- metrics for the dashboard ----
    y_nn = np_p.argmax(1)
    tr, va = subject_split(X, y, s)
    val_nn = numpy_predict(layers, X.iloc[va].to_numpy()).argmax(1)
    metrics = {
        "classes": list(le.classes_),
        "dataset": {
            "train_windows": int(len(X)), "test_windows": int(len(Xt)), "features": int(X.shape[1]),
            "train_subjects": int(len(set(s))), "test_subjects": int(len(set(st))),
            "train_counts": np.bincount(y, minlength=len(le.classes_)).tolist(),
        },
        "keras": {
            "test_accuracy": float(accuracy_score(yt, y_nn)),
            "val_accuracy": float(accuracy_score(y[va], val_nn)),
            "macro_f1": float(f1_score(yt, y_nn, average="macro")),
            "f1": f1_score(yt, y_nn, average=None).round(4).tolist(),
            "confusion": confusion_matrix(yt, y_nn).tolist(),
            "params": int(model.count_params()),
            "architecture": "561 → 512 → 256 → 128 → 6 (Dense + BatchNorm + ReLU + Dropout)",
        },
    }
    rf_path = art / "rf_baseline.joblib"
    if rf_path.exists():
        rf = joblib.load(rf_path)["model"]
        y_rf = rf.predict(Xt)
        metrics["random_forest"] = {
            "test_accuracy": float(accuracy_score(yt, y_rf)),
            "macro_f1": float(f1_score(yt, y_rf, average="macro")),
            "f1": f1_score(yt, y_rf, average=None).round(4).tolist(),
        }
    if args.history and Path(args.history).exists():
        h = json.load(open(args.history))
        metrics["keras"]["history"] = h
        metrics["keras"]["best_epoch"] = int(np.argmin(h["val_loss"])) + 1
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # ---- real test windows for the live demo (stratified, 100 per class) ----
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(yt == c)[0], 100, replace=False)
                          for c in range(len(le.classes_))])
    rng.shuffle(idx)
    np.savez_compressed(out / "samples.npz", X=Xt.to_numpy()[idx].astype(np.float32),
                        y=yt[idx].astype(np.int8), subject=st[idx].astype(np.int16))
    print("Wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
