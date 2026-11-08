"""Random Forest baseline for UCI HAR."""
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from har_load import load_har, subject_split

data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
X_train, y_train, s_train, X_test, y_test, s_test, le = load_har(data_dir)


def make_rf():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)


# 1) Subject-wise validation (people unseen during training)
tr, va = subject_split(X_train, y_train, s_train)
rf = make_rf().fit(X_train.iloc[tr], y_train[tr])
print(f"Validation accuracy (unseen subjects): {accuracy_score(y_train[va], rf.predict(X_train.iloc[va])):.4f}")

# 2) Retrain on all training data, evaluate on test
rf = make_rf().fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

# 3) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title(f"Random Forest - Test confusion matrix (acc {accuracy_score(y_test, y_pred):.3f})")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=150)
print("Saved rf_confusion_matrix.png")

# Save model + label encoder for later (e.g. FastAPI)
joblib.dump({"model": rf, "label_encoder": le}, "rf_baseline.joblib")
print("Saved rf_baseline.joblib")
