"""Step 1 - Load and preprocess the customer data for K-Means clustering."""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/churn-bigml-80.csv"
OUT_DIR = "data/processed"
SCALER_PATH = "artifacts/scaler.joblib"

# Dropped columns and why:
CHARGE_COLS = ["Total day charge", "Total eve charge",
               "Total night charge", "Total intl charge"]   # exact copies of the minutes columns
ID_COLS = ["State", "Area code"]                            # identifiers, not behaviour
LABEL_COL = "Churn"                                         # kept aside for interpretation only
REDUNDANT = ["Voice mail plan"]                             # same information as 'Number vmail messages'


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def build_features(df):
    """Keep only customer behaviour columns and turn Yes/No into 1/0."""
    features = df.drop(columns=CHARGE_COLS + ID_COLS + REDUNDANT + [LABEL_COL]).copy()
    features["International plan"] = (features["International plan"] == "Yes").astype(int)
    return features


def scale_features(features):
    """Standardize every feature (mean 0, std 1). K-Means uses distances, so a
    column measured in the hundreds (minutes) would otherwise swamp one
    measured in single digits (service calls)."""
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(features),
                          columns=features.columns, index=features.index)
    return scaled, scaler


def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} customers, {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())} | Duplicates: {int(df.duplicated().sum())}\n")

    # The voice mail plan flag is fully explained by the message count
    assert ((df["Voice mail plan"] == "Yes") == (df["Number vmail messages"] > 0)).all()
    features = build_features(df)
    scaled, scaler = scale_features(features)

    print(f"Features used ({features.shape[1]}): {list(features.columns)}")
    print(f"Dropped: {len(CHARGE_COLS)} charge columns (duplicates of minutes), "
          f"{ID_COLS}, 'Voice mail plan'; '{LABEL_COL}' is held out\n")

    # Checks
    assert features.isna().sum().sum() == 0
    assert not any(c in features.columns for c in CHARGE_COLS + [LABEL_COL])
    assert scaled.mean().abs().max() < 1e-9, "Scaled mean is not ~0"
    assert (scaled.std(ddof=0) - 1).abs().max() < 1e-9, "Scaled std is not ~1"
    print("Preprocessing verified")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    features.assign(Churn=df[LABEL_COL]).to_csv(f"{OUT_DIR}/customers_features.csv", index=False)
    scaled.to_csv(f"{OUT_DIR}/customers_scaled.csv", index=False)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved -> {OUT_DIR}/customers_features.csv, customers_scaled.csv and {SCALER_PATH}")


if __name__ == "__main__":
    main()
