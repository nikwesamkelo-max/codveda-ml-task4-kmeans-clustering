"""Step 3 - Visualize the customer clusters with 2D scatter plots."""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

SCALED_PATH = "data/processed/customers_scaled.csv"
CLUSTERED_PATH = "data/processed/customers_clustered.csv"
MODEL_PATH = "artifacts/kmeans_model.joblib"
CHART_PATH = "images/cluster_scatter.png"
PALETTE = ["#2563eb", "#f97316", "#16a34a", "#9333ea"]
RANDOM_STATE = 42


def project_to_2d(X_scaled, centers):
    """Squash the 12 features down to 2 axes (PCA) that keep as much of the
    variation as possible, so the clusters can be drawn on a flat chart."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_scaled)
    points = pd.DataFrame(pca.transform(X_scaled), columns=["PC1", "PC2"])
    centers = pd.DataFrame(centers, columns=X_scaled.columns)  # keep feature names
    centroids = pd.DataFrame(pca.transform(centers), columns=["PC1", "PC2"])
    return points, centroids, pca.explained_variance_ratio_


def plot_clusters(points, centroids, variance, clustered, labels):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    palette = PALETTE[: len(centroids)]

    # Left: PCA view of all 12 features
    ax = axes[0]
    sns.scatterplot(x=points["PC1"], y=points["PC2"], hue=labels, palette=palette,
                    s=22, alpha=0.6, linewidth=0, ax=ax, legend=False)
    ax.scatter(centroids["PC1"], centroids["PC2"], c=palette, marker="X", s=380,
               edgecolor="black", linewidth=1.8, zorder=5)
    ax.set_xlabel(f"Principal component 1 ({variance[0]:.0%} of variation)")
    ax.set_ylabel(f"Principal component 2 ({variance[1]:.0%} of variation)")
    ax.set_title("All 12 features squashed to 2D", fontsize=16, weight="bold")

    # Right: two real features
    ax = axes[1]
    sns.scatterplot(data=clustered, x="Account length", y="Number vmail messages",
                    hue="cluster", palette=palette, s=22, alpha=0.6, linewidth=0, ax=ax)
    means = clustered.groupby("cluster")[["Account length", "Number vmail messages"]].mean()
    ax.scatter(means["Account length"], means["Number vmail messages"], c=palette,
               marker="X", s=380, edgecolor="black", linewidth=1.8, zorder=5)
    ax.set_title("Two real features", fontsize=16, weight="bold")
    ax.legend(title="Cluster", frameon=False, loc="upper right")

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Customer clusters found by K-Means (X = cluster centre)",
                 fontsize=20, weight="bold")
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    X = pd.read_csv(SCALED_PATH)
    clustered = pd.read_csv(CLUSTERED_PATH)
    model = joblib.load(MODEL_PATH)

    points, centroids, variance = project_to_2d(X, model.cluster_centers_)
    print(f"2D view keeps {variance.sum():.1%} of the variation "
          f"(PC1 {variance[0]:.1%}, PC2 {variance[1]:.1%})")

    # Checks
    assert len(points) == len(clustered) == len(X)
    assert len(centroids) == model.n_clusters
    assert set(clustered["cluster"]) == set(range(model.n_clusters))
    plot_clusters(points, centroids, variance, clustered, clustered["cluster"])
    assert os.path.getsize(CHART_PATH) > 0
    print("Visualization verified")
    print(f"Saved -> {CHART_PATH}")


if __name__ == "__main__":
    main()
