"""PhishGuard training pipeline.

Run from the project root:
    python -m src.train            # full study (about 10-25 minutes)
    python -m src.train --fast     # quick demo run (about 2-4 minutes)

Study design (no data leakage):
  1. Remove exact duplicate rows, then make ONE stratified 80/20 train/test split.
  2. Compare 11 models x 3 feature conditions with 5-fold stratified CV on the TRAIN part only.
     Scaler / PCA / SelectKBest live inside each pipeline, so they are re-fit on every fold.
  3. Pick the best configuration by CV accuracy, tune it (train part only), refit.
  4. Evaluate the chosen model ONCE on the untouched test part.
"""
import argparse
import json
import time
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
                             f1_score, make_scorer, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import CLASS_LABELS, FEATURES, ROOT, TARGET, load_data

warnings.filterwarnings("ignore")
SEED = 42
RESULTS, FIGS, MODELS = ROOT / "results", ROOT / "results" / "figures", ROOT / "models"
CONDITIONS = ["Original 30", "PCA 95%", "SelectKBest 15"]
PHISH = 0  # encoded label of the phishing class (the class we most want to catch)

BASE_MODELS = ["Logistic Regression", "KNN", "SVM RBF", "Decision Tree", "Gaussian NB"]
ENSEMBLE_MODELS = ["Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "Soft Voting", "Stacking"]


def build_models():
    lr = lambda: LogisticRegression(max_iter=2000, random_state=SEED)
    return {
        "Logistic Regression": lr(),
        "KNN": KNeighborsClassifier(),
        "SVM RBF": SVC(kernel="rbf", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1, verbosity=0),
        "Soft Voting": VotingClassifier(
            [("lr", lr()), ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)), ("nb", GaussianNB())],
            voting="soft"),
        "Stacking": StackingClassifier(
            [("lr", lr()), ("knn", KNeighborsClassifier()), ("nb", GaussianNB()),
             ("dt", DecisionTreeClassifier(random_state=SEED))],
            final_estimator=LogisticRegression(max_iter=2000), cv=3),
    }


GRIDS = {
    "Random Forest": {"clf__n_estimators": [200, 300, 500], "clf__max_depth": [None, 15, 25],
                      "clf__min_samples_split": [2, 4, 8], "clf__max_features": ["sqrt", "log2"]},
    "XGBoost": {"clf__n_estimators": [150, 250, 400], "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.8, 1.0]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 5],
                          "clf__learning_rate": [0.05, 0.1, 0.2]},
    "SVM RBF": {"clf__C": [0.5, 1, 3, 10, 30], "clf__gamma": ["scale", 0.01, 0.03, 0.1]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 11, 15], "clf__weights": ["uniform", "distance"]},
    "Logistic Regression": {"clf__C": [0.1, 1, 10, 100]},
    "Decision Tree": {"clf__max_depth": [5, 8, 12, None], "clf__min_samples_leaf": [1, 3, 5]},
    "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.3, 0.5, 1.0]},
}


def make_pipe(model, condition):
    steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if condition == "PCA 95%":
        steps += [("scale", StandardScaler()), ("reduce", PCA(n_components=0.95, svd_solver="full"))]
    elif condition == "SelectKBest 15":
        steps += [("select", SelectKBest(mutual_info_classif, k=15)), ("scale", StandardScaler())]
    else:
        steps += [("scale", StandardScaler())]
    return Pipeline(steps + [("clf", clone(model))])


def n_features_used(fitted, condition):
    if condition == "PCA 95%":
        return int(fitted.named_steps["reduce"].n_components_)
    return 15 if condition == "SelectKBest 15" else 30


def phish_scores(y_true, y_pred):
    return {"precision_phish": precision_score(y_true, y_pred, pos_label=PHISH, zero_division=0),
            "recall_phish": recall_score(y_true, y_pred, pos_label=PHISH, zero_division=0),
            "f1_phish": f1_score(y_true, y_pred, pos_label=PHISH, zero_division=0)}


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)


def main(fast=False):
    t_start = time.perf_counter()
    for folder in (RESULTS, FIGS, MODELS):
        folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    # ------------------------------------------------------------------ data
    raw = load_data()
    if raw.shape[1] != 31 or TARGET not in raw:
        raise ValueError(f"Expected 30 features + {TARGET}, got {raw.shape}")
    dup_count = int(raw.duplicated().sum())
    df = raw.drop_duplicates().reset_index(drop=True)
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])  # -1 -> 0 (phishing), 1 -> 1 (legitimate)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    n_splits = 3 if fast else 5
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    info = {"raw_rows": len(raw), "duplicate_rows_removed": dup_count, "rows_used": len(df),
            "features": len(FEATURES), "phishing_rows": int((y == 0).sum()), "legitimate_rows": int((y == 1).sum()),
            "train_rows": len(y_train), "test_rows": len(y_test), "cv_folds": n_splits, "fast_mode": fast,
            "sklearn_version": sklearn.__version__, "trained_at": datetime.now().isoformat(timespec="seconds")}
    (RESULTS / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"Rows: raw={len(raw)}, duplicates removed={dup_count}, used={len(df)} | train={len(y_train)} test={len(y_test)}")

    # ------------------------------------------- stage 1: fair CV comparison
    models = build_models()
    if fast:
        models = {k: models[k] for k in ["Logistic Regression", "Decision Tree", "Gaussian NB", "Random Forest", "XGBoost"]}
    scoring = {"accuracy": "accuracy", "precision": "precision_weighted", "recall": "recall_weighted",
               "f1": "f1_weighted", "roc_auc": "roc_auc",
               "precision_phish": make_scorer(precision_score, pos_label=PHISH, zero_division=0),
               "recall_phish": make_scorer(recall_score, pos_label=PHISH, zero_division=0),
               "f1_phish": make_scorer(f1_score, pos_label=PHISH, zero_division=0)}
    rows, fold_rows = [], []
    for name, estimator in models.items():
        for cond in CONDITIONS:
            print(f"[CV] {name:<20} | {cond}", flush=True)
            pipe = make_pipe(estimator, cond)
            t0 = time.perf_counter()
            sc = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            cv_time = time.perf_counter() - t0
            fitted = clone(pipe)
            t0 = time.perf_counter()
            fitted.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            row = {"model": name, "family": "Ensemble" if name in ENSEMBLE_MODELS else "Base", "condition": cond,
                   "n_features_used": n_features_used(fitted, cond)}
            for key in scoring:
                row[f"cv_{key}_mean"] = float(np.mean(sc[f"test_{key}"]))
            row["cv_accuracy_std"] = float(np.std(sc["test_accuracy"], ddof=1))
            row["cv_time_sec"], row["fit_time_sec"] = cv_time, fit_time
            rows.append(row)
            for i in range(n_splits):
                fold_rows.append({"model": name, "condition": cond, "fold": i + 1,
                                  "accuracy": sc["test_accuracy"][i], "f1_phish": sc["test_f1_phish"][i]})
    summary = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", summary.index + 1)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "cv_folds.csv", index=False)

    # ------------------------------- stage 2: tune ONLY the winner (train data)
    best = summary.iloc[0]
    best_name, best_cond = best["model"], best["condition"]
    best_pipe = make_pipe(models[best_name], best_cond)
    tuning = {"model": best_name, "condition": best_cond, "default_cv_accuracy": float(best["cv_accuracy_mean"]),
              "tuned": False, "best_params": {}, "tuned_inner_cv_accuracy": None}
    if best_name in GRIDS:
        grid = GRIDS[best_name]
        space = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(best_pipe, grid, n_iter=min(4 if fast else 12, space), scoring="accuracy",
                                    cv=StratifiedKFold(3, shuffle=True, random_state=SEED), random_state=SEED,
                                    n_jobs=1, refit=True)
        search.fit(X_train, y_train)
        tuning["tuned_inner_cv_accuracy"] = float(search.best_score_)
        if search.best_score_ >= best["cv_accuracy_mean"]:
            best_pipe, tuning["tuned"] = search.best_estimator_, True
            tuning["best_params"] = {k: (v if isinstance(v, (int, float, str)) or v is None else str(v))
                                     for k, v in search.best_params_.items()}
    final = clone(best_pipe)
    if isinstance(final.named_steps["clf"], SVC):
        final.set_params(clf__probability=True)  # needed for probabilities in the app
    final.fit(X_train, y_train)
    (RESULTS / "tuning.json").write_text(json.dumps(tuning, indent=2))

    # ---------------------------------- stage 3: one-time test-set evaluation
    p_legit = final.predict_proba(X_test)[:, 1]
    pred = final.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    test = {"selected_model": best_name, "feature_condition": best_cond, "tuned": tuning["tuned"],
            "selection_metric": f"{n_splits}-fold CV accuracy on training data",
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "test_recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "test_f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, p_legit), **{f"test_{k}": v for k, v in phish_scores(y_test, pred).items()},
            "phish_caught": int(tn), "phish_missed": int(fn), "legit_blocked": int(fp), "legit_passed": int(tp),
            "test_rows": len(y_test), "duplicate_rows_removed_before_split": dup_count}
    (RESULTS / "final_test_metrics.json").write_text(json.dumps(test, indent=2))
    pd.DataFrame({"y_true": y_test, "p_legit": p_legit, "y_pred": pred}, index=X_test.index).join(X_test).to_csv(
        RESULTS / "test_predictions.csv", index_label="row_id")

    joblib.dump(final, MODELS / "best_model.joblib")
    meta = {"model": best_name, "condition": best_cond, "feature_names": FEATURES, "tuned": tuning["tuned"],
            "best_params": tuning["best_params"], "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
            "original_classes": [str(c) for c in le.classes_], "sklearn_version": sklearn.__version__,
            "selection_metric": test["selection_metric"]}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSelected: {best_name} | {best_cond} | tuned={tuning['tuned']}")
    print(f"Held-out TEST accuracy = {test['test_accuracy']:.4f}, phishing recall = {test['test_recall_phish']:.4f}")

    # ------------------------------------------------ explanations & PCA data
    imp = permutation_importance(final, X_test, y_test, n_repeats=5, random_state=SEED, scoring="accuracy", n_jobs=1)
    perm = pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std}
                        ).sort_values("importance", ascending=False)
    perm.to_csv(RESULTS / "permutation_importance.csv", index=False)

    scaler = StandardScaler().fit(X_train)  # fit on training data only
    pca = PCA().fit(scaler.transform(X_train))
    evr = pca.explained_variance_ratio_
    n95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1)
    pd.DataFrame({"component": np.arange(1, 31), "explained_variance_ratio": evr, "cumulative": np.cumsum(evr)}
                 ).to_csv(RESULTS / "pca_variance.csv", index=False)
    load = pd.DataFrame(pca.components_[:5].T, index=FEATURES, columns=[f"PC{i}" for i in range(1, 6)])
    load.to_csv(RESULTS / "pca_loadings.csv", index_label="feature")
    proj = pca.transform(scaler.transform(X))[:, :3]
    pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "PC3": proj[:, 2],
                  "class": [CLASS_LABELS[v] for v in y],
                  "split": np.where(X.index.isin(X_train.index), "train", "test")}).round(4).to_csv(
        RESULTS / "pca_projection.csv", index=False)
    X.corr().round(3).to_csv(RESULTS / "feature_correlation.csv")

    # PCA component sweep for the best base model and best ensemble model (by CV)
    sweep_rows = []
    pick = [summary[summary.family == "Base"].iloc[0]["model"], summary[summary.family == "Ensemble"].iloc[0]["model"]]
    ks = [2, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30] if fast else [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30]
    for name in dict.fromkeys(pick):
        for k in ks:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("scale", StandardScaler()),
                             ("pca", PCA(n_components=k)), ("clf", clone(models[name]))])
            s = cross_validate(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)["test_score"]
            sweep_rows.append({"model": name, "n_components": k, "cv_accuracy_mean": s.mean(),
                               "cv_accuracy_std": s.std(ddof=1)})
        print(f"[PCA sweep] {name} done", flush=True)
    pd.DataFrame(sweep_rows).to_csv(RESULTS / "pca_component_sweep.csv", index=False)

    # ------------------------------------------------------ report figures
    pivot = summary.pivot(index="model", columns="condition", values="cv_accuracy_mean")[CONDITIONS]
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={"label": "CV accuracy"})
    ax.set(title="Cross-validated accuracy: model x feature condition", xlabel="", ylabel="")
    save_fig(fig, "accuracy_heatmap.png")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.barplot(data=summary, x="model", y="cv_f1_phish_mean", hue="condition", ax=ax, order=pivot.index)
    ax.set(title="Cross-validated F1 for the phishing class", xlabel="", ylabel="F1 (phishing)", ylim=(0.5, 1))
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "f1_comparison.png")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Phishing", "Legitimate"], ax=ax, cmap="Blues")
    ax.set_title(f"Held-out test confusion matrix\n{best_name} ({best_cond})")
    save_fig(fig, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    RocCurveDisplay.from_predictions(y_test, p_legit, ax=ax, name=best_name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("ROC curve (held-out test set)")
    save_fig(fig, "roc_selected_model.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(1, 31), evr, color="#2a9d8f", label="Per component")
    ax.plot(np.arange(1, 31), np.cumsum(evr), "o-", color="#264653", label="Cumulative")
    ax.axhline(0.95, ls="--", color="grey")
    ax.axvline(n95, ls=":", color="crimson", label=f"95% at {n95} components")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio", title="PCA scree + cumulative variance")
    ax.legend()
    save_fig(fig, "pca_scree.png")

    sw = pd.read_csv(RESULTS / "pca_component_sweep.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for name, g in sw.groupby("model"):
        ax.errorbar(g.n_components, g.cv_accuracy_mean, yerr=g.cv_accuracy_std, marker="o", capsize=3, label=name)
    ax.set(xlabel="PCA components kept", ylabel="CV accuracy", title="How many components are enough?")
    ax.legend()
    save_fig(fig, "accuracy_vs_pca_components.png")

    top = perm.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#e76f51")
    ax.set(title="Permutation importance (drop in test accuracy when a feature is shuffled)", xlabel="Accuracy drop")
    ax.title.set_fontsize(10)
    save_fig(fig, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Phishing", "Legitimate"], [(y == 0).sum(), (y == 1).sum()], color=["#e63946", "#2a9d8f"])
    ax.set(title="Class balance after removing duplicates", ylabel="Rows")
    save_fig(fig, "class_balance.png")

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    save_fig(fig, "correlation_heatmap.png")

    minutes = (time.perf_counter() - t_start) / 60
    info["training_minutes"] = round(minutes, 1)
    (RESULTS / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nTraining complete in {minutes:.1f} min. Launch the app:  streamlit run app.py")


if __name__ == "__main__":  # the guard is required on Windows (joblib / multiprocessing)
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="smaller, quicker study for demos")
    main(fast=parser.parse_args().fast)
