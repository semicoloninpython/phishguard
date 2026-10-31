"""PhishGuard - Streamlit dashboard.   Run:  streamlit run app.py"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import ui
from src.features import url_to_uci_features
from src.url_scanner import inspect_url
from src.utils import FEATURES, FEATURE_INFO, ROOT, URL_ONLY_FEATURES, get_split, title_of

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="wide")
ui.inject_css()
RES, MODEL_FILE, META_FILE = ROOT / "results", ROOT / "models" / "best_model.joblib", ROOT / "models" / "metadata.json"
MEANING = {-1: "Phishing-like", 0: "Suspicious", 1: "Legitimate-like"}
RED, GREEN = ui.PALETTE["phish"], ui.PALETTE["legit"]


@st.cache_data
def read_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_json(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_FILE), json.loads(META_FILE.read_text())
    except Exception:
        return None, None


@st.cache_data
def train_part():
    return get_split()[0::2]  # X_train, y_train


def goto(name):
    st.session_state["nav"] = name


summary, info, tm, tune = read_csv("summary.csv"), read_json("results/dataset_info.json"), read_json("results/final_test_metrics.json"), read_json("results/tuning.json")
model, meta = load_model()
READY = summary is not None and tm is not None and model is not None


# ============================================================ OVERVIEW
def overview():
    ui.hero("COMPARATIVE ML STUDY · UCI PHISHING WEBSITES",
            "PhishGuard: spotting phishing websites with PCA and ensemble learning",
            "A phishing page copies a trusted site to steal passwords. We describe every website with 30 measurable "
            "signals, then compare <b>11 classifiers</b> on <b>3 feature strategies</b> (all features, PCA, SelectKBest) "
            "to find out which combination detects phishing best, and whether dimensionality reduction and "
            "ensembles really help.")
    if not READY:
        st.warning("Models are not trained yet. Run `python -m src.train` in the project folder, then refresh this page.")
        return
    best = summary.iloc[0]
    c = st.columns(4)
    with c[0]: ui.card("Websites analysed", f"{info['rows_used']:,}", f"{info['duplicate_rows_removed']:,} duplicates removed · 30 features")
    with c[1]: ui.card("Experiments", len(summary), f"{summary.model.nunique()} models × 3 feature strategies")
    with c[2]: ui.card("Best CV accuracy", f"{best.cv_accuracy_mean:.1%}", f"{best.model} · {best.condition}", "good")
    with c[3]: ui.card("Held-out test accuracy", f"{tm['test_accuracy']:.1%}", f"Phishing recall {tm['test_recall_phish']:.1%} · seen once", "good")
    ui.section("How the study works", "Everything below is reproducible with <code>python -m src.train</code>.")
    ui.flow([("Clean", "Remove exact duplicate rows"), ("Split", "80% train / 20% locked test set"),
             ("Transform", "Scale, then PCA or SelectKBest"), ("Compare", "11 models, 5-fold CV"),
             ("Select", "Best CV accuracy wins; tune it"), ("Test", "One-time check on unseen data")])
    ui.section("What the results say")
    piv = summary.pivot(index="model", columns="condition", values="cv_accuracy_mean")
    bb, be = summary[summary.family == "Base"].iloc[0], summary[summary.family == "Ensemble"].iloc[0]
    pca_delta = (piv["PCA 95%"] - piv["Original 30"]) * 100
    var = read_csv("pca_variance.csv")
    n95 = int((var.cumulative < 0.95).sum() + 1) if var is not None else None
    ui.insight(f"🏆 <b>{best.model}</b> on <b>{best.condition}</b> leads the leaderboard with {best.cv_accuracy_mean:.2%} CV accuracy.")
    ui.insight(f"🌲 Best ensemble (<b>{be.model}</b>, {be.cv_accuracy_mean:.2%}) vs best single model (<b>{bb.model}</b>, {bb.cv_accuracy_mean:.2%}): "
               f"<b>{(be.cv_accuracy_mean - bb.cv_accuracy_mean) * 100:+.2f} points</b>.")
    ui.insight(f"📉 PCA changed accuracy by <b>{pca_delta.mean():+.2f} points on average</b> vs all 30 features; it helped {int((pca_delta > 0).sum())} "
               f"of {len(pca_delta)} models" + (f" and needs {n95} components to keep 95% of the variance." if n95 else "."))
    ui.insight(f"🧹 {info['duplicate_rows_removed']:,} exact duplicates ({info['duplicate_rows_removed'] / info['raw_rows']:.0%} of the raw file) were removed "
               "before splitting. Otherwise identical rows land in both train and test and inflate accuracy.")
    st.write("")
    c = st.columns(4)
    for col, (label, target) in zip(c, [("📚 Understand the concept", "📚  Understand"), ("🏁 Open the leaderboard", "🏁  Leaderboard"),
                                        ("🧪 Try the live predictor", "🧪  Live Predictor"), ("🔎 Scan a URL", "🔎  URL Scanner")]):
        col.button(label, on_click=goto, args=(target,), width="stretch", key=f"cta_{target}")


# ============================================================ UNDERSTAND
def understand():
    ui.header("CONCEPT GUIDE", "Understand the project in 5 minutes", "Plain-language answers to: what, why, how, and what to say in the viva.")
    t = st.tabs(["1 · The problem", "2 · The data", "3 · PCA", "4 · Ensembles", "5 · Viva Q&A"])
    with t[0]:
        st.markdown("""
**Phishing** is a fake website (bank, e-mail, shop) built to steal logins or card numbers. Blacklists are always late because
new fake sites appear every hour, so researchers ask: *can a model learn the typical **fingerprints** of a fake site?*

**Our idea:** describe each website by 30 signals (is the URL very long? is there an `@`? how old is the domain? does the form submit
to another site?). Each signal is coded **-1** (phishing-like), **0** (suspicious) or **1** (legitimate-like). A classifier learns which
combinations of signals point to phishing.
""")
        ui.flow([("Website", "URL + page + domain"), ("30 features", "coded -1 / 0 / 1"), ("Preprocess", "scale · PCA · select"),
                 ("Classifier", "single model or ensemble"), ("Verdict", "phishing or legitimate")])
        ui.insight("<b>Why the title says comparative analysis:</b> there is no single best model. We measure accuracy, phishing-class recall, F1 and "
                   "ROC-AUC for every model × feature strategy and let the numbers decide.")
    with t[1]:
        meta_df = pd.DataFrame([{"Feature": k, "Meaning": v[0], "Group": v[1], "Computed from": v[2], "What it checks": v[3]} for k, v in FEATURE_INFO.items()])
        c1, c2 = st.columns([1, 1.4])
        counts = meta_df["Computed from"].value_counts().reset_index()
        counts.columns = ["source", "features"]
        fig = px.bar(counts, x="features", y="source", orientation="h", title="Where each of the 30 features comes from",
                     color_discrete_sequence=[ui.PALETTE["accent"]])
        fig.update_yaxes(categoryorder="total ascending")
        c1.plotly_chart(ui.style_fig(fig, 430), width="stretch")
        with c2:
            ui.insight(f"Only <b>{len(URL_ONLY_FEATURES)} of 30</b> features can be read from the URL text. The other {30 - len(URL_ONLY_FEATURES)} need the page HTML, "
                       "WHOIS/DNS or web-ranking services, so a trained model cannot judge a raw URL on its own. That is why the Live Predictor "
                       "uses dataset feature vectors and the URL Scanner is a separate rule-based tool.")
            ui.insight("<b>Encoding:</b> -1 = phishing-like · 0 = suspicious · 1 = legitimate-like. Target <code>Result</code>: -1 phishing, 1 legitimate.")
        group = st.multiselect("Filter by group", sorted(meta_df.Group.unique()), default=sorted(meta_df.Group.unique()))
        st.dataframe(meta_df[meta_df.Group.isin(group)], width="stretch", hide_index=True, height=360)
    with t[2]:
        st.markdown("""
**Problem:** 30 features are many, partly redundant (e.g. several domain-reputation signals move together), and slow for distance-based models.

**PCA (Principal Component Analysis)** rotates the data to new axes called *components*. The 1st component is the direction with the most variance, the 2nd is the next
one (perpendicular to the 1st), and so on. We keep the first *k* components that hold **95 % of the variance**. Think of photographing a 3-D object from the angle that shows
the most detail.

**Catch:** PCA maximises *variance*, not *class separation*. With ternary (-1/0/1) features it may drop a small signal that was the best phishing clue. That is why we **measure** instead of assuming PCA helps.

**Leakage rule:** the scaler and PCA are fitted **inside each CV fold** (a scikit-learn `Pipeline`), never on the test data.
""")
        st.info("Open the **PCA Lab** page to see the variance curve, the 2-D projection and to train live with any number of components.")
    with t[3]:
        st.dataframe(pd.DataFrame([
            ["Bagging", "Random Forest", "Many trees on random samples; vote. Cuts variance."],
            ["Boosting", "AdaBoost · Gradient Boosting · XGBoost", "Trees added one by one, each fixing the previous errors. Cuts bias."],
            ["Voting", "Soft Voting (LR + RF + NB)", "Average the predicted probabilities of different models."],
            ["Stacking", "Stacking (LR, KNN, NB, DT → LR)", "A meta-model learns how to combine the base models."]],
            columns=["Idea", "Models here", "Intuition"]), width="stretch", hide_index=True)
        st.markdown("**Base models** for comparison: Logistic Regression, KNN, SVM (RBF), Decision Tree, Gaussian Naive Bayes.")
    with t[4]:
        nb = ""
        if summary is not None:
            p = summary.pivot(index="model", columns="condition", values="cv_accuracy_mean")
            if "Gaussian NB" in p.index:
                nb = f" In our run Gaussian NB scored {p.loc['Gaussian NB', 'Original 30']:.1%} on raw features and {p.loc['Gaussian NB', 'PCA 95%']:.1%} after PCA."
        qa = {
            "Why did you remove duplicate rows?": "Identical rows can appear in both train and test, so the model is graded on data it has memorised. We remove exact duplicates before the split.",
            "Why use cross-validation AND a test set?": "CV on the training part compares and selects models. The test set is touched once, at the end, to give an unbiased estimate of real performance.",
            "Why is the scaler/PCA inside a Pipeline?": "So they are re-fitted on each training fold only. Fitting PCA on all data would leak information from validation/test rows.",
            "Does PCA always improve results?": "No. PCA keeps variance, not class information. It helps models hurt by correlated features and slow distance calculations, and can hurt tree models." + nb,
            "Bagging vs boosting vs stacking?": "Bagging trains independent models on resamples and averages them (less variance). Boosting trains sequentially on previous errors (less bias). Stacking learns a meta-model on top of base models.",
            "Why is phishing recall more important than accuracy?": "A missed phishing site (false negative) can cost a user their account; a false alarm only costs a click. Hence we report precision/recall/F1 for the phishing class and offer a threshold slider.",
            "What is ROC-AUC?": "The probability that a random phishing site gets a higher phishing score than a random legitimate one. 0.5 = chance, 1.0 = perfect ranking.",
            "What are the limitations?": "The dataset is old (2012 era), features need crawling/WHOIS, attackers adapt (concept drift), and the URL Scanner is only a heuristic. Treat PhishGuard as an academic demonstration."}
        for q, a in qa.items():
            with st.expander(q):
                st.write(a)


# ============================================================ LEADERBOARD
def leaderboard():
    ui.header("MODEL COMPARISON", "Leaderboard: 11 models × 3 feature strategies", "All numbers are 5-fold cross-validation on the training partition (no test data used).")
    if summary is None:
        st.warning("Run the training pipeline first."); return
    metrics = {"Accuracy": "cv_accuracy_mean", "F1 · phishing class": "cv_f1_phish_mean", "Recall · phishing class": "cv_recall_phish_mean",
               "Precision · phishing class": "cv_precision_phish_mean", "ROC-AUC": "cv_roc_auc_mean", "Weighted F1": "cv_f1_mean"}
    c1, c2, c3 = st.columns([1, 1, 1.2])
    fam = c1.multiselect("Model family", ["Base", "Ensemble"], default=["Base", "Ensemble"])
    cond = c2.multiselect("Feature strategy", list(summary.condition.unique()), default=list(summary.condition.unique()))
    mlabel = c3.selectbox("Metric", list(metrics))
    col = metrics[mlabel]
    d = summary[summary.family.isin(fam) & summary.condition.isin(cond)].copy()
    if d.empty:
        st.info("Select at least one family and strategy."); return
    top = d.sort_values(col, ascending=False).iloc[0]
    k = st.columns(3)
    with k[0]: ui.card(f"Best by {mlabel}", f"{top[col]:.3f}", f"{top.model} · {top.condition}", "good")
    with k[1]: ui.card("Spread (best − worst)", f"{(d[col].max() - d[col].min()) * 100:.1f} pts", f"across {len(d)} configurations")
    fam_mean = d.groupby("family")[col].mean()
    with k[2]: ui.card("Mean Base vs Ensemble", " vs ".join(f"{fam_mean.get(f, float('nan')):.3f}" for f in ["Base", "Ensemble"]), "average over the selection")
    piv = d.pivot(index="model", columns="condition", values=col)
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(piv, text_auto=".3f", aspect="auto", color_continuous_scale="Viridis", title=f"{mlabel}: model × feature strategy")
    st.plotly_chart(ui.style_fig(fig, 460), width="stretch")
    a, b = st.columns(2)
    fig = px.bar(d.sort_values(col, ascending=False), x="model", y=col, color="condition", barmode="group", error_y="cv_accuracy_std" if col == "cv_accuracy_mean" else None,
                 title=f"{mlabel} by model")
    fig.update_yaxes(range=[max(0.5, d[col].min() - 0.03), 1.0])
    a.plotly_chart(ui.style_fig(fig, 420), width="stretch")
    fig = px.scatter(d, x="fit_time_sec", y=col, color="family", symbol="condition", hover_data=["model", "condition"], log_x=True,
                     title="Accuracy vs training time (log scale)", color_discrete_map={"Base": ui.PALETTE["accent"], "Ensemble": ui.PALETTE["warn"]})
    fig.update_traces(marker=dict(size=13))
    b.plotly_chart(ui.style_fig(fig, 420), width="stretch")
    folds = read_csv("cv_folds.csv")
    if folds is not None:
        folds["config"] = folds.model + " · " + folds.condition
        top8 = d.sort_values("cv_accuracy_mean", ascending=False).head(8)
        keep = (top8.model + " · " + top8.condition).tolist()
        fig = px.box(folds[folds.config.isin(keep)], x="config", y="accuracy", points="all", title="Fold-to-fold stability of the top 8 configurations")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(ui.style_fig(fig, 440), width="stretch")
    ui.section("Full results table")
    show = d.sort_values("cv_accuracy_mean", ascending=False)[["rank", "model", "family", "condition", "n_features_used", "cv_accuracy_mean", "cv_accuracy_std",
                                                                "cv_precision_phish_mean", "cv_recall_phish_mean", "cv_f1_phish_mean", "cv_roc_auc_mean", "fit_time_sec"]]
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "cv_accuracy_mean": st.column_config.ProgressColumn("CV accuracy", min_value=0.5, max_value=1.0, format="%.3f"),
        "cv_accuracy_std": st.column_config.NumberColumn("± std", format="%.4f"),
        "cv_precision_phish_mean": st.column_config.NumberColumn("Precision (phish)", format="%.3f"),
        "cv_recall_phish_mean": st.column_config.NumberColumn("Recall (phish)", format="%.3f"),
        "cv_f1_phish_mean": st.column_config.NumberColumn("F1 (phish)", format="%.3f"),
        "cv_roc_auc_mean": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
        "fit_time_sec": st.column_config.NumberColumn("Fit time (s)", format="%.2f"), "n_features_used": "Features used"})
    st.download_button("⬇ Download full summary (CSV)", summary.to_csv(index=False).encode(), "phishguard_summary.csv", "text/csv")


# ============================================================ PCA LAB
ZOO = {"Logistic Regression": lambda: LogisticRegression(max_iter=2000), "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
       "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42), "XGBoost": lambda: XGBClassifier(eval_metric="logloss", n_jobs=1, verbosity=0, random_state=42),
       "KNN": lambda: KNeighborsClassifier(), "Gaussian NB": lambda: GaussianNB()}


@st.cache_data(show_spinner=False)
def live_cv(name, k):
    X, y = train_part()
    steps = [("imp", SimpleImputer(strategy="most_frequent")), ("sc", StandardScaler())]
    if k < 30:
        steps.append(("pca", PCA(n_components=k)))
    t0 = time.perf_counter()
    s = cross_val_score(Pipeline(steps + [("clf", ZOO[name]())]), X, y, cv=StratifiedKFold(3, shuffle=True, random_state=42), n_jobs=1)
    return float(s.mean()), float(s.std(ddof=1)), time.perf_counter() - t0


def pca_lab():
    ui.header("DIMENSIONALITY REDUCTION", "PCA Lab", "See how much information each component keeps and how accuracy changes when you drop dimensions.")
    var, proj, load, sweep = read_csv("pca_variance.csv"), read_csv("pca_projection.csv"), read_csv("pca_loadings.csv"), read_csv("pca_component_sweep.csv")
    if var is None:
        st.warning("Run the training pipeline first."); return
    target = st.slider("Variance to keep", 0.50, 0.99, 0.95, 0.01, format="%.2f")
    n_need = int((var.cumulative < target).sum() + 1)
    k = st.columns(3)
    with k[0]: ui.card("Components needed", f"{n_need} / 30", f"to keep {target:.0%} of the variance")
    with k[1]: ui.card("Dimensions removed", f"{30 - n_need}", f"{(30 - n_need) / 30:.0%} fewer features", "good")
    with k[2]: ui.card("PC1 holds", f"{var.explained_variance_ratio.iloc[0]:.1%}", "the most variance of any direction")
    fig = go.Figure()
    fig.add_bar(x=var.component, y=var.explained_variance_ratio, name="Per component", marker_color=ui.PALETTE["accent"])
    fig.add_scatter(x=var.component, y=var.cumulative, name="Cumulative", mode="lines+markers", line=dict(color=ui.PALETTE["warn"]))
    fig.add_hline(y=target, line_dash="dash", line_color="#94a3b8")
    fig.add_vline(x=n_need, line_dash="dot", line_color=RED)
    fig.update_layout(title="Scree plot and cumulative explained variance", xaxis_title="Principal component", yaxis_title="Explained variance")
    st.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    a, b = st.columns(2)
    if proj is not None:
        dim = a.radio("Projection", ["2D", "3D"], horizontal=True)
        sample = proj.sample(min(2500, len(proj)), random_state=1)
        cmap = {"Phishing": RED, "Legitimate": GREEN}
        fig = px.scatter_3d(sample, x="PC1", y="PC2", z="PC3", color="class", color_discrete_map=cmap, opacity=0.6) if dim == "3D" else \
            px.scatter(sample, x="PC1", y="PC2", color="class", color_discrete_map=cmap, opacity=0.5)
        fig.update_layout(title="Websites projected onto the first principal components")
        a.plotly_chart(ui.style_fig(fig, 470), width="stretch")
    if load is not None:
        pc = b.selectbox("Which component?", ["PC1", "PC2", "PC3", "PC4", "PC5"])
        top = load.assign(a=load[pc].abs(), name=load.feature.map(title_of)).sort_values("a", ascending=False).head(10).iloc[::-1]
        fig = px.bar(top, x=pc, y="name", orientation="h", title=f"Features that drive {pc} (loadings)", color=pc, color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        b.plotly_chart(ui.style_fig(fig, 410), width="stretch")
    if sweep is not None:
        fig = px.line(sweep, x="n_components", y="cv_accuracy_mean", color="model", markers=True, error_y="cv_accuracy_std", title="Accuracy vs number of PCA components (from training run)",
                      labels={"n_components": "Components kept", "cv_accuracy_mean": "CV accuracy"})
        st.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    ui.section("Train it live", "Pick a model and a number of components. The app runs 3-fold CV on the training data right now.")
    c1, c2 = st.columns(2)
    name, comps = c1.selectbox("Model", list(ZOO), index=2), c2.slider("PCA components", 2, 30, n_need)
    if st.button("▶ Run experiment", type="primary"):
        with st.spinner("Cross-validating..."):
            (m, s, t), (m0, s0, t0) = live_cv(name, comps), live_cv(name, 30)
        r = st.columns(4)
        r[0].metric(f"{name} · PCA {comps}", f"{m:.2%}", f"{(m - m0) * 100:+.2f} pts vs all 30")
        r[1].metric("All 30 features", f"{m0:.2%}")
        r[2].metric("Run time (3-fold)", f"{t:.1f}s", f"{t - t0:+.1f}s vs all 30", delta_color="inverse")
        r[3].metric("Fold std", f"±{s:.3f}")
        ui.insight("Remember: PCA keeps variance, not necessarily class information. A negative delta is a valid scientific finding.")


# ============================================================ TEST RESULTS
def results():
    ui.header("FINAL EVALUATION", "Held-out test results", "The selected model was evaluated exactly once on data it never saw during training or model selection.")
    tp = read_csv("test_predictions.csv")
    if not READY or tp is None:
        st.warning("Run the training pipeline first."); return
    k = st.columns(5)
    for col, (l, v, s, tone) in zip(k, [("Model", tm["selected_model"], tm["feature_condition"] + (" · tuned" if tm["tuned"] else ""), ""),
                                       ("Accuracy", f"{tm['test_accuracy']:.2%}", f"{tm['test_rows']} unseen websites", "good"),
                                       ("Phishing recall", f"{tm['test_recall_phish']:.2%}", "share of phishing sites caught", "good"),
                                       ("Phishing precision", f"{tm['test_precision_phish']:.2%}", "alerts that were truly phishing", ""),
                                       ("ROC-AUC", f"{tm['test_roc_auc']:.3f}", "ranking quality", "")]):
        with col: ui.card(l, v, s, tone)
    p_phish, y_true = 1 - tp.p_legit.to_numpy(), (tp.y_true.to_numpy() == 0).astype(int)  # 1 = phishing
    ui.section("Threshold lab", "Flag a site as phishing when P(phishing) ≥ threshold. Lower it to catch more attacks at the cost of more false alarms.")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    pred = (p_phish >= thr).astype(int)
    TP, FP, FN, TN = int(((pred == 1) & (y_true == 1)).sum()), int(((pred == 1) & (y_true == 0)).sum()), int(((pred == 0) & (y_true == 1)).sum()), int(((pred == 0) & (y_true == 0)).sum())
    prec, rec = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    r = st.columns(4)
    r[0].metric("Phishing caught", f"{TP}", f"{FN} missed", delta_color="inverse")
    r[1].metric("False alarms", f"{FP}", f"{TN} legit passed")
    r[2].metric("Recall / Precision", f"{rec:.1%} / {prec:.1%}")
    r[3].metric("F1 (phishing)", f"{f1:.3f}", f"accuracy {(TP + TN) / len(y_true):.1%}")
    a, b = st.columns(2)
    cm = np.array([[TP, FN], [FP, TN]])
    fig = px.imshow(cm, text_auto=True, x=["Predicted phishing", "Predicted legitimate"], y=["Actually phishing", "Actually legitimate"], color_continuous_scale="Blues", title="Confusion matrix at this threshold")
    a.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    fpr, tpr, _ = roc_curve(y_true, p_phish)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc(fpr, tpr):.3f})", line=dict(color=ui.PALETTE["accent"], width=3))
    fig.add_scatter(x=[1 - TN / max(TN + FP, 1) if (TN + FP) else 0], y=[rec], mode="markers", name="Current threshold", marker=dict(size=14, color=RED))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#64748b"), name="Chance")
    fig.update_layout(title="ROC curve", xaxis_title="False-alarm rate", yaxis_title="Phishing recall")
    b.plotly_chart(ui.style_fig(fig, 400), width="stretch")
    ts = np.linspace(0.05, 0.95, 91)
    pr = pd.DataFrame({"t": ts, "Precision": [((p_phish >= t) & (y_true == 1)).sum() / max((p_phish >= t).sum(), 1) for t in ts],
                       "Recall": [((p_phish >= t) & (y_true == 1)).sum() / max(y_true.sum(), 1) for t in ts]})
    fig = px.line(pr.melt("t"), x="t", y="value", color="variable", title="Precision and recall as the threshold moves", labels={"t": "Threshold", "value": ""})
    fig.add_vline(x=thr, line_dash="dot", line_color=RED)
    st.plotly_chart(ui.style_fig(fig, 340), width="stretch")
    perm = read_csv("permutation_importance.csv")
    if perm is not None:
        top = perm.head(12).iloc[::-1].assign(name=lambda d: d.feature.map(title_of))
        fig = px.bar(top, x="importance", y="name", orientation="h", error_x="std", title="Which features matter most? (permutation importance on test data)",
                     labels={"importance": "Accuracy drop when shuffled", "name": ""}, color_discrete_sequence=[ui.PALETTE["violet"]])
        st.plotly_chart(ui.style_fig(fig, 460), width="stretch")
    if tune:
        with st.expander("Tuning details"):
            st.json(tune)


# ============================================================ LIVE PREDICTOR
def predict_frame(rows):
    p_legit = model.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]
    return 1 - p_legit


def gauge(p):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}}, title={"text": "P(phishing)"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": RED if p >= .5 else GREEN},
                                        "steps": [{"range": [0, 35], "color": "rgba(52,211,153,.18)"}, {"range": [35, 65], "color": "rgba(251,191,36,.18)"}, {"range": [65, 100], "color": "rgba(248,113,113,.2)"}]}))
    return ui.style_fig(fig, 300)


def live_predictor():
    ui.header("MODEL LAB", "Live predictor", "Pick a real website from the held-out test set, change its signals and watch the model react.")
    tp = read_csv("test_predictions.csv")
    if not READY or tp is None:
        st.warning("Run the training pipeline first."); return
    tabs = st.tabs(["🎯 Sample explorer", "📄 Batch CSV"])
    with tabs[0]:
        ui.insight("The 30 features come from the page, WHOIS and DNS, so we start from <b>real test-set websites</b> instead of a typed URL. Edit any signal to run a what-if.")
        c1, c2 = st.columns([1, 1])
        which = c1.radio("Show me", ["Any", "Phishing", "Legitimate"], horizontal=True)
        pool = tp if which == "Any" else tp[tp.y_true == (0 if which == "Phishing" else 1)]
        if st.session_state.get("pool_key") != which or "sid" not in st.session_state:
            st.session_state.update(pool_key=which, sid=int(pool.row_id.sample(1, random_state=int(time.time()) % 10000).iloc[0]))
        if c2.button("🎲 New random website", width="stretch"):
            st.session_state["sid"] = int(pool.row_id.sample(1).iloc[0])
        sid = st.session_state["sid"]
        row = tp[tp.row_id == sid].iloc[0]
        vals, sample_name = {}, f"Test website #{sid}"
        st.markdown(f"**{sample_name}** · dataset label: **{'Phishing' if row.y_true == 0 else 'Legitimate'}**")
        groups = {}
        for f in FEATURES:
            groups.setdefault(FEATURE_INFO[f][1], []).append(f)
        for g, feats in groups.items():
            with st.expander(f"{g} ({len(feats)} features)", expanded=(g == "Address bar")):
                cols = st.columns(3)
                for i, f in enumerate(feats):
                    vals[f] = cols[i % 3].selectbox(title_of(f), [-1, 0, 1], index=[-1, 0, 1].index(int(row[f])), key=f"f_{sid}_{f}", help=FEATURE_INFO[f][3], format_func=lambda v: f"{v} · {MEANING[v]}")
        p = float(predict_frame([vals])[0])
        left, right = st.columns([1, 1.2])
        left.plotly_chart(gauge(p), width="stretch")
        verdict = "Phishing website" if p >= .5 else "Legitimate website"
        with right:
            st.markdown(f'<div class="pg-verdict {"danger" if p >= .5 else "safe"}"><div class="pg-kicker">MODEL VERDICT · {meta["model"]}</div><h2>{verdict}</h2>'
                        f'<p>P(phishing) = {p:.1%}. {"Matches" if (p >= .5) == (row.y_true == 0) else "Differs from"} the dataset label for the original sample.</p></div>', unsafe_allow_html=True)
        base = [dict(vals)] + [{**vals, f: 0} for f in FEATURES]
        probs = predict_frame(base)
        contrib = pd.DataFrame({"feature": [title_of(f) for f in FEATURES], "shift": probs[0] - probs[1:]})
        contrib = contrib.reindex(contrib["shift"].abs().sort_values(ascending=False).index).head(8).iloc[::-1]
        fig = px.bar(contrib, x="shift", y="feature", orientation="h", color=contrib["shift"] > 0, color_discrete_map={True: RED, False: GREEN},
                     title="Why? Change in P(phishing) when each feature is set to 'unknown' (0)", labels={"shift": "pushes toward phishing →", "feature": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(ui.style_fig(fig, 380), width="stretch")
    with tabs[1]:
        st.caption("Upload a CSV with all 30 UCI feature columns (values -1, 0, 1). A 'Result' column is ignored. Maximum 5,000 rows.")
        sample = tp[FEATURES].head(10)
        st.download_button("⬇ Download a sample CSV", sample.to_csv(index=False).encode(), "phishguard_sample.csv", "text/csv")
        file = st.file_uploader("Feature CSV", type=["csv"])
        if file:
            try:
                inc = pd.read_csv(file)
                missing = [f for f in FEATURES if f not in inc]
                if missing:
                    st.error("Missing columns: " + ", ".join(missing))
                elif not inc[FEATURES].isin([-1, 0, 1]).all().all():
                    st.error("All feature values must be -1, 0 or 1.")
                elif len(inc) > 5000:
                    st.error("Maximum 5,000 rows per upload.")
                else:
                    out = inc[FEATURES].copy()
                    pp = predict_frame(out)
                    out["P(phishing)"], out["Prediction"] = pp.round(4), np.where(pp >= .5, "Phishing", "Legitimate")
                    st.metric("Flagged as phishing", f"{(pp >= .5).sum()} of {len(out)}")
                    st.dataframe(out, width="stretch", height=380)
                    st.download_button("⬇ Download predictions", out.to_csv(index=False).encode(), "phishguard_predictions.csv", "text/csv")
            except Exception as exc:
                st.error(f"Could not process this CSV: {exc}")


# ============================================================ URL SCANNER
def scanner():
    ui.header("QUICK CHECK · RULE-BASED · NO API KEY", "URL scanner", "Inspects the visible URL text only. It does not open the website and is <b>not</b> the machine-learning model.")
    with st.form("scan"):
        url = st.text_input("Website URL", value=st.session_state.get("scan_url", ""), placeholder="https://example.com/login")
        go_btn = st.form_submit_button("🔎 Scan URL", type="primary", width="stretch")
    if go_btn:
        st.session_state["scan_url"], st.session_state["scan_res"] = url, inspect_url(url)
    res = st.session_state.get("scan_res")
    if not res:
        examples = ["https://www.wikipedia.org", "http://192.168.10.5:8080/secure-login/verify@account", "https://paypal-secure-update.account-verify.example.xyz/login"]
        st.caption("Try an example: " + " · ".join(f"`{e}`" for e in examples))
        return
    if not res["valid"]:
        st.error(res["error"]); return
    a, b = st.columns([1, 1.6])
    fig = go.Figure(go.Indicator(mode="gauge+number", value=res["score"], title={"text": "URL risk indicator"}, number={"suffix": "/100"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": {"safe": GREEN, "warning": ui.PALETTE["warn"], "danger": RED}[res["status"]]},
                                        "steps": [{"range": [0, 18], "color": "rgba(52,211,153,.15)"}, {"range": [18, 40], "color": "rgba(251,191,36,.15)"}, {"range": [40, 100], "color": "rgba(248,113,113,.17)"}]}))
    a.plotly_chart(ui.style_fig(fig, 320), width="stretch")
    b.markdown(f'<div class="pg-verdict {res["status"]}"><div class="pg-kicker">URL-ONLY HEURISTICS</div><h2>{res["verdict"]}</h2><p>{res["explanation"]}</p>'
               f'<p style="font-size:13px;color:#a8bad0">Host inspected: <b>{res["host"]}</b></p></div>', unsafe_allow_html=True)
    ui.section("What we noticed")
    for f in res["findings"]:
        st.markdown(f'<div class="pg-find"><b>{f["title"]}</b> <span class="pg-pill {"ok" if f["severity"] == "safe" else "no"}">{f["points"]} pts</span><p>{f["detail"]}</p></div>', unsafe_allow_html=True)
    ui.section("Link to the dataset", f"The same URL expressed as the {len(URL_ONLY_FEATURES)} UCI features that can be computed from text alone.")
    feats = url_to_uci_features(res["url"])
    st.dataframe(pd.DataFrame([{"Feature": title_of(k), "Value": v, "Reading": MEANING[v]} for k, v in feats.items()]), width="stretch", hide_index=True)
    with st.expander(f"The other {30 - len(URL_ONLY_FEATURES)} features this scanner cannot see"):
        st.dataframe(pd.DataFrame([{"Feature": v[0], "Needs": v[2]} for k, v in FEATURE_INFO.items() if k not in URL_ONLY_FEATURES]), width="stretch", hide_index=True)
    st.warning("A low score is not proof of safety: the scanner does not fetch the page, follow redirects, check domain age or query threat intelligence.")


# ============================================================ ABOUT
def about():
    ui.header("PROJECT INFORMATION", "About, method and limits")
    if info:
        c = st.columns(4)
        for col, (l, v) in zip(c, [("Raw rows", f"{info['raw_rows']:,}"), ("Duplicates removed", f"{info['duplicate_rows_removed']:,}"), ("Train / test", f"{info['train_rows']:,} / {info['test_rows']:,}"), ("CV folds", info["cv_folds"])]):
            with col: ui.card(l, v)
    st.markdown("""
**Title:** Comparative Analysis of Machine Learning Models for Phishing Website Detection using Dimensionality Reduction and Ensemble Learning.

**Dataset:** UCI *Website Phishing / Phishing Websites* (ID 327), Mohammad, Thabtah & McCluskey (2012-2015): 30 engineered features, labels -1 phishing / 1 legitimate.

**Method:** duplicate removal → stratified 80/20 split → pipelines (imputer, scaler, optional PCA 95 % or SelectKBest-15) → 11 classifiers → 5-fold CV → best configuration tuned → one-time test evaluation.

**Reproducibility:** fixed seed 42; all artefacts (summary, folds, PCA data, test predictions) are written to `results/`.

**Limitations**
- Features need crawling/WHOIS, so the ML model cannot score a bare URL.
- The data is dated and from one collection effort; real attackers change tactics (concept drift).
- PCA maximises variance, not class separation; discrete features may not benefit.
- The URL Scanner is a rule-based heuristic and gives no probability.
- Educational demonstration, not a standalone security control.
""")
    if info:
        st.caption(f"Trained {info.get('trained_at', '')} with scikit-learn {info.get('sklearn_version', '')} · fast mode: {info.get('fast_mode')}")
    for f, label in [("summary.csv", "experiment summary"), ("final_test_metrics.json", "final test metrics")]:
        p = RES / f
        if p.exists():
            st.download_button(f"⬇ Download {label}", p.read_bytes(), f)


# ============================================================ ROUTER
PAGES = {"🏠  Overview": overview, "📚  Understand": understand, "🏁  Leaderboard": leaderboard, "🧬  PCA Lab": pca_lab,
         "📊  Test Results": results, "🧪  Live Predictor": live_predictor, "🔎  URL Scanner": scanner, "ℹ️  About": about}
st.session_state.setdefault("nav", "🏠  Overview")
with st.sidebar:
    st.markdown('<div class="pg-brand">🛡️ Phish<span>Guard</span></div>', unsafe_allow_html=True)
    st.caption("PHISHING DETECTION · ML RESEARCH")
    st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
    st.divider()
    st.caption(("● " if READY else "○ ") + ("Model ready: " + meta["model"] if READY else "Model not trained"))
    st.caption("UCI Phishing Websites · ID 327")
PAGES[st.session_state["nav"]]()
