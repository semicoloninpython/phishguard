# PhishGuard

**Comparative Analysis of Machine Learning Models for Phishing Website Detection using Dimensionality Reduction and Ensemble Learning**

11 classifiers (5 base + 6 ensemble) x 3 feature strategies (all 30 features, PCA 95%, SelectKBest 15) on the UCI Phishing Websites dataset, with a Streamlit dashboard.

## Run it (Windows, PowerShell or CMD, Python 3.10+)

```powershell
cd phishguard
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m src.train          # trains everything, ~5-25 min (use --fast for a 2-3 min demo run)
streamlit run app.py         # opens http://localhost:8501
```
Shortcuts: `setup.bat` (install) and `run_app.bat` (start the app).

## Run it (macOS / Linux)
```bash
./setup.sh && source .venv/bin/activate
python -m src.train && streamlit run app.py
```

Trained results are already included in `results/` and `models/`, so you can run `streamlit run app.py` straight after installing. Re-run `python -m src.train` to reproduce them (the data is in `data/phishing.csv`; `python fetch_data.py` re-downloads it from UCI).

## Notebooks (optional)
`jupyter notebook notebooks/` and run 1 to 5 in order (they read `results/`).

## Project layout
```
app.py              dashboard (8 pages)
src/train.py        full study: CV comparison, tuning, test, PCA data, figures
src/features.py     URL text -> the 9 UCI features that can be derived from a URL
src/url_scanner.py  rule-based URL heuristics (not ML)
src/utils.py        paths, data loading, feature descriptions
notebooks/          5 executed notebooks with observations
results/            summary.csv, cv_folds.csv, final_test_metrics.json, figures/ ...
models/             best_model.joblib + metadata.json
docs/VIVA_GUIDE.md  concept summary and viva answers
archive_old_run/    the original, leakage-affected run (kept for reference only)
```

## Method (no leakage)
1. Remove 5,206 exact duplicate rows, then one stratified 80/20 split (seed 42).
2. 5-fold CV on the training part; scaler / PCA / SelectKBest are inside each fold's Pipeline.
3. Select best by CV accuracy, tune it on training data only, evaluate once on the test set.
4. Report phishing-class precision, recall and F1, not just accuracy.

## Honest limitations
- 21 of the 30 features need the page HTML, WHOIS/DNS or ranking services, so the ML model scores **dataset feature vectors**, not raw URLs. The URL Scanner is a separate heuristic.
- Dataset is dated; attackers change tactics. Educational demo, not a security product.
