"""Stage 1 helpers: data loading, preprocessing, models and CV evaluation.
All fitting happens on training data only (inside each CV fold)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from config import SEED, DATA

XLSX = DATA / "bank_campaign_participant_release.xlsx"
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def load_stage1():
    train = pd.read_excel(XLSX, sheet_name="Stage1_Train")
    test = pd.read_excel(XLSX, sheet_name="Stage1_Test")
    X = train[NUMERIC + CATEGORICAL]
    y = (train["y"] == "yes").astype(int).to_numpy()
    return train, test, X, y


def _add_features(df):
    df = df.copy()
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].clip(lower=0)
    df["balance"] = np.sign(df["balance"]) * np.log1p(
        np.abs(df["balance"])
    )  # tame heavy tail
    return df


def make_preprocessor():
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC + ["was_contacted_before"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("feat", FunctionTransformer(_add_features)), ("ct", ct)])


def classical_model(C=0.3, class_weight=None):
    return Pipeline(
        [
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000)),
        ]
    )


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """Classical preprocessing -> PCA to n_qubits -> angle scaling ->
    ZZFeatureMap fidelity kernel (statevector, exact) -> SVC with Platt scaling."""

    def __init__(self, n_qubits=4, reps=1, C=1.0, scale=0.5, reducer="pca", seed=SEED):
        self.n_qubits, self.reps, self.C, self.scale, self.reducer, self.seed = (
            n_qubits,
            reps,
            C,
            scale,
            reducer,
            seed,
        )

    def _kernel(self):
        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.kernels import FidelityStatevectorKernel

        return FidelityStatevectorKernel(
            feature_map=zz_feature_map(self.n_qubits, reps=self.reps)
        )

    def fit(self, X, y):
        if self.reducer == "pca":
            red = PCA(n_components=self.n_qubits, random_state=self.seed)
        else:  # supervised: keep the n_qubits features with largest L1-logistic weights (fit on train fold only)
            from sklearn.feature_selection import SelectFromModel

            red = SelectFromModel(
                LogisticRegression(
                    penalty="l1", solver="liblinear", C=0.5, random_state=self.seed
                ),
                max_features=self.n_qubits,
                threshold=-np.inf,
            )
        self.prep_ = Pipeline(
            [
                ("prep", make_preprocessor()),
                ("reduce", red),
                ("angle", StandardScaler()),
            ]
        ).fit(X, y)
        Z = self._angles(self.prep_.transform(X))
        self.Z_train_ = Z
        self.kernel_ = self._kernel()
        K = self.kernel_.evaluate(Z)
        self.svc_ = SVC(
            kernel="precomputed", C=self.C, probability=True, random_state=self.seed
        ).fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X):
        Z = self._angles(self.prep_.transform(X))
        return self.svc_.predict_proba(self.kernel_.evaluate(Z, self.Z_train_))

    def _angles(self, Z):
        # smaller scale = wider kernel (smoother), avoids the kernel collapsing to identity
        return np.clip(Z, -3, 3) * self.scale

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def precision_recall_at_k(y_true, p, k=10):
    top = np.argsort(-p, kind="stable")[:k]
    hits = y_true[top].sum()
    return hits / k, hits / max(y_true.sum(), 1)


def cross_validate(model, X, y, n_splits=5, seed=SEED):
    """Stratified CV on training data only. Returns per-fold metrics and out-of-fold probabilities."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = clone(model).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        p10, r10 = precision_recall_at_k(y[va], p, 10)
        rows.append(
            dict(
                fold=fold,
                roc_auc=roc_auc_score(y[va], p),
                pr_auc=average_precision_score(y[va], p),
                brier=brier_score_loss(y[va], p),
                log_loss=log_loss(y[va], p, labels=[0, 1]),
                precision_at_10=p10,
                recall_at_10=r10,
            )
        )
    return pd.DataFrame(rows), oof
