"""Load the UCI HAR (Kaggle) dataset: encode labels, keep subject for splitting."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


def load_har(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Label-encode Activity (fit on train only, reuse on test)
    le = LabelEncoder()
    y_train = le.fit_transform(train["Activity"])
    y_test = le.transform(test["Activity"])

    # Keep subject separately for subject-wise splits
    subj_train = train["subject"].to_numpy()
    subj_test = test["subject"].to_numpy()

    # Features = everything except subject and Activity
    X_train = train.drop(columns=["subject", "Activity"])
    X_test = test.drop(columns=["subject", "Activity"])

    return X_train, y_train, subj_train, X_test, y_test, subj_test, le


def subject_split(X, y, groups, val_size=0.2, seed=42):
    """Validation split where no subject appears in both train and val."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, va_idx


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    X_train, y_train, s_train, X_test, y_test, s_test, le = load_har(data_dir)

    print("X_train:", X_train.shape, "| y_train:", y_train.shape)
    print("X_test: ", X_test.shape, "| y_test: ", y_test.shape)
    print("Missing values (train/test):", int(X_train.isna().sum().sum()), "/", int(X_test.isna().sum().sum()))

    print("\nLabel mapping:")
    for i, name in enumerate(le.classes_):
        print(f"  {i} = {name}")

    print("\nClass counts (train):")
    print(pd.Series(le.inverse_transform(y_train)).value_counts().to_string())
    print("\nClass counts (test):")
    print(pd.Series(le.inverse_transform(y_test)).value_counts().to_string())

    print("\nSubjects: train", len(set(s_train)), "| test", len(set(s_test)),
          "| overlap", len(set(s_train) & set(s_test)))

    tr, va = subject_split(X_train, y_train, s_train)
    print(f"\nSubject-wise val split: train={len(tr)}, val={len(va)}, "
          f"subject overlap={len(set(s_train[tr]) & set(s_train[va]))}")
