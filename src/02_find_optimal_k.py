"""Step 2 - Apply K-Means and find the optimal number of clusters (elbow method)."""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SCALED_PATH = "data/processed/customers_scaled.csv"
FEATURES_PATH = "data/processed/customers_features.csv"
CLUSTERED_PATH = "data/processed/customers_clustered.csv"
MODEL_PATH = "artifacts/kmeans_model.joblib"
RESULTS_PATH = "outputs/k_selection.csv"
K_PATH = "outputs/optimal_k.json"
CHART_PATH = "images/elbow_method.png"
K_RANGE = range(1, 11)
RANDOM_STATE = 42


def evaluate_k_values(X):
    """Fit K-Means for each k. Inertia = total squared distance from every
    customer to the centre of its cluster (lower = tighter clusters)."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        silhouette = silhouette_score(X, km.labels_) if k > 1 else np.nan
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette})
    return pd.DataFrame(rows)


def find_elbow(results):
    """Elbow = the point farthest from the straight line joining the first and
    last points of the inertia curve (where adding clusters stops paying off)."""
    k = results["k"].to_numpy(dtype=float)
    y = results["inertia"].to_numpy()
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    distance = np.abs((kn - kn[0]) * (yn[-1] - yn[0]) - (yn - yn[0]) * (kn[-1] - kn[0]))
    return int(k[distance.argmax()])


def plot_selection(results, k_elbow):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].plot(results["k"], results["inertia"], "o-", color="#2563eb", lw=2.5, ms=8)
    axes[0].axvline(k_elbow, color="#16a34a", ls="--", lw=2)
    axes[0].set_title(f"Elbow method: k = {k_elbow}", fontsize=17, weight="bold")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (lower = tighter clusters)")
    s = results.dropna()
    axes[1].plot(s["k"], s["silhouette"], "o-", color="#f97316", lw=2.5, ms=8)
    axes[1].set_title("Silhouette score (cross-check)", fontsize=17, weight="bold")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette (higher = better separated)")
    for ax in axes:
        ax.set_xticks(list(K_RANGE))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    X = pd.read_csv(SCALED_PATH)
    results = evaluate_k_values(X)
    k_elbow = find_elbow(results)
    k_silhouette = int(results.dropna().loc[results["silhouette"].idxmax(), "k"])

    print(results.round(3).to_string(index=False))
    print(f"\nElbow method suggests k = {k_elbow}")
    print(f"Best silhouette score is at k = {k_silhouette} "
          f"({results.loc[results['k'] == k_silhouette, 'silhouette'].iloc[0]:.3f})")
    print(f"Silhouette at k = {k_elbow}: "
          f"{results.loc[results['k'] == k_elbow, 'silhouette'].iloc[0]:.3f} "
          "(low = clusters overlap; see interpretation in step 4)\n")

    # Apply K-Means with the elbow choice
    model = KMeans(n_clusters=k_elbow, n_init=10, random_state=RANDOM_STATE).fit(X)
    sizes = pd.Series(model.labels_).value_counts().sort_index()
    print("Cluster sizes:", sizes.to_dict())

    # Checks
    assert results["inertia"].is_monotonic_decreasing, "Inertia should fall as k grows"
    assert model.n_clusters == k_elbow and len(model.labels_) == len(X)
    assert (sizes > 0).all() and len(sizes) == k_elbow
    again = KMeans(n_clusters=k_elbow, n_init=10, random_state=RANDOM_STATE).fit(X)
    assert (again.labels_ == model.labels_).all(), "Results are not reproducible"
    print("Clustering verified (reproducible)")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    results.round(4).to_csv(RESULTS_PATH, index=False)
    with open(K_PATH, "w") as f:
        json.dump({"k_elbow": k_elbow, "k_best_silhouette": k_silhouette}, f, indent=2)
    joblib.dump(model, MODEL_PATH)
    pd.read_csv(FEATURES_PATH).assign(cluster=model.labels_).to_csv(CLUSTERED_PATH, index=False)
    plot_selection(results, k_elbow)
    print(f"Saved -> {MODEL_PATH}, {CLUSTERED_PATH}, {RESULTS_PATH}, {K_PATH}, {CHART_PATH}")


if __name__ == "__main__":
    main()
