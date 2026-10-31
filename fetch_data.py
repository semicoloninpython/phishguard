from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Fetching UCI Phishing Websites dataset (id=327)...")
phishing = fetch_ucirepo(id=327)

X = phishing.data.features.copy()
y = phishing.data.targets.copy()

if isinstance(y, pd.DataFrame):
    if y.shape[1] != 1:
        raise ValueError(f"Expected one target column, found {y.shape[1]}")
    y = y.iloc[:, 0]

df = X.copy()
df["Result"] = y.to_numpy()

if X.shape[1] != 30:
    raise ValueError(f"Expected 30 features, received {X.shape[1]}. Check the UCI dataset schema.")

out = DATA_DIR / "phishing.csv"
df.to_csv(out, index=False)

# reload the saved file to verify the actual artifact
check = pd.read_csv(out)
print("Saved:", out)
print("Shape:", check.shape)
print("Features:", len(check.columns) - 1)
print("Missing values:", int(check.isna().sum().sum()))
print("Target distribution:")
print(check["Result"].value_counts(dropna=False))
print("Duplicate rows:", int(check.duplicated().sum()))
