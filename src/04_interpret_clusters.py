"""Step 4 - Interpret the clusters: name the segments and compare their churn."""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLUSTERED_PATH = "data/processed/customers_clustered.csv"
MODEL_PATH = "artifacts/kmeans_model.joblib"
SCALER_PATH = "artifacts/scaler.joblib"
HOLDOUT_PATH = "data/raw/churn-bigml-20.csv"   # optional: customers the model never saw
PROFILE_PATH = "outputs/segment_profiles.csv"
CHURN_PATH = "outputs/churn_by_segment.csv"
HEATMAP_PATH = "images/segment_profile_heatmap.png"
CHURN_CHART_PATH = "images/churn_by_segment.png"

# Same columns that were dropped in step 1
DROP_COLS = ["Total day charge", "Total eve charge", "Total night charge", "Total intl charge",
             "State", "Area code", "Voice mail plan", "Churn"]


def name_clusters(profile, overall):
    """Give each cluster a plain-English name from its average behaviour."""
    names = {}
    for cluster, row in profile.iterrows():
        if row["International plan"] > 0.5:
            names[cluster] = "International plan holders"
        elif row["Number vmail messages"] > 15:
            names[cluster] = "Voicemail users"
        elif row["Account length"] < overall["Account length"]:
            names[cluster] = "Newer accounts"
        else:
            names[cluster] = "Established accounts"
    return names


def build_profile(clustered, feature_cols):
    profile = clustered.groupby("cluster")[feature_cols].mean()
    profile.insert(0, "customers", clustered.groupby("cluster").size())
    profile.insert(1, "share_%", (profile["customers"] / len(clustered) * 100))
    profile["churn_%"] = clustered.groupby("cluster")["Churn"].mean() * 100
    return profile


def assign_holdout(path, model, scaler, feature_cols):
    """Place never-seen customers into the existing clusters (no re-fitting)."""
    raw = pd.read_csv(path)
    feats = raw.drop(columns=DROP_COLS)
    feats["International plan"] = (feats["International plan"] == "Yes").astype(int)
    feats = feats[feature_cols]
    return raw.assign(cluster=model.predict(pd.DataFrame(scaler.transform(feats), columns=feature_cols)))


def plot_heatmap(model, feature_cols, names):
    centers = pd.DataFrame(model.cluster_centers_, columns=feature_cols,
                           index=[names[i] for i in range(len(names))])
    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.heatmap(centers, annot=True, fmt=".1f", cmap="coolwarm", center=0, vmin=-3, vmax=3,
                linewidths=1, cbar_kws={"label": "Standard deviations from the average customer"}, ax=ax)
    ax.set_title("What makes each segment different", fontsize=19, weight="bold", loc="left")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    os.makedirs("images", exist_ok=True)
    fig.savefig(HEATMAP_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_churn(churn):
    fig, ax = plt.subplots(figsize=(12, 6.75))
    churn.plot.bar(ax=ax, color=["#2563eb", "#f97316"][: churn.shape[1]], width=0.75)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", fontsize=13, padding=3)
    ax.set_ylabel("Customers who churned (%)")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_title("Churn rate by customer segment", fontsize=20, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(CHURN_CHART_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    clustered = pd.read_csv(CLUSTERED_PATH)
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = [c for c in clustered.columns if c not in ("Churn", "cluster")]

    profile = build_profile(clustered, feature_cols)
    overall = clustered[feature_cols].mean()
    names = name_clusters(profile, overall)
    assert len(set(names.values())) == len(names), "Two clusters got the same name"
    profile.index = [names[i] for i in profile.index]

    show = ["customers", "share_%", "Account length", "International plan",
            "Number vmail messages", "Customer service calls", "churn_%"]
    print("Segment profiles:\n")
    print(profile[show].round(1).to_string())
    print(f"\nOverall churn: {clustered['Churn'].mean():.1%}\n")

    churn = pd.DataFrame({"Training customers": profile["churn_%"]})
    if os.path.exists(HOLDOUT_PATH):
        holdout = assign_holdout(HOLDOUT_PATH, model, scaler, feature_cols)
        held = holdout.groupby("cluster")["Churn"].mean() * 100
        churn["Unseen customers"] = [held[i] for i in range(len(names))]
        print(f"Same clusters applied to {len(holdout)} unseen customers "
              f"(overall churn {holdout['Churn'].mean():.1%}):")
    else:
        print("(Optional check skipped: data/raw/churn-bigml-20.csv not found)")
    print(churn.round(1).to_string(), "\n")

    # Checks
    assert profile["customers"].sum() == len(clustered)
    assert churn.min().min() >= 0 and churn.max().max() <= 100
    top = profile["International plan"].idxmax()
    assert top == "International plan holders"
    print("Interpretation verified")

    os.makedirs("outputs", exist_ok=True)
    profile.round(2).to_csv(PROFILE_PATH)
    churn.round(2).to_csv(CHURN_PATH)
    plot_heatmap(model, feature_cols, names)
    plot_churn(churn.loc[profile.sort_values("churn_%", ascending=False).index])
    print(f"Saved -> {PROFILE_PATH}, {CHURN_PATH}, {HEATMAP_PATH}, {CHURN_CHART_PATH}")


if __name__ == "__main__":
    main()
