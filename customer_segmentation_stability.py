"""
Stability-Aware Customer Segmentation
=====================================
Selects customer segments using BOTH cluster quality (compactness/separation)
and cluster stability under resampling.

Pipeline
  1. Build / load customer RFM data and preprocess
  2. Define candidate models (K-Means, Ward, Gaussian Mixture, DBSCAN)
  3. Score quality (silhouette, Calinski-Harabasz, Davies-Bouldin)
  4. Score stability by re-clustering perturbed data (80% subsamples and noise
     injection) and measuring Adjusted Rand Index (ARI) vs the reference clustering
  5. Combine quality and stability into one score (harmonic mean) + parsimony rule
  6. Check weight sensitivity and per-segment (Jaccard) stability
  7. Profile the final segments

Usage
  python customer_segmentation_stability.py                       # synthetic data
  python customer_segmentation_stability.py --data customers.csv  # your own data
  python customer_segmentation_stability.py --data customers.csv --B 100 --out results

Your CSV must contain the columns: recency, frequency, monetary.
Outputs (tables as CSV, plots as PNG) are written to the --out directory.
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend: works on servers with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------- Configuration ----------------
SEED = 42
FEATURES = ["recency", "frequency", "monetary"]
SUBSAMPLE_FRAC = 0.8              # fraction kept in each subsample
NOISE_SD = 0.15                   # Gaussian noise (in standardized units)
K_RANGE = range(2, 9)
DBSCAN_EPS = [0.3, 0.5, 0.7]
DBSCAN_MIN_SAMPLES = 20
PARSIMONY_TOL = 0.02              # prefer fewer clusters if score is within this of the best
JACCARD_THRESHOLD = 0.75          # segments below this are flagged unreliable


def parse_args():
    p = argparse.ArgumentParser(description="Stability-aware customer segmentation")
    p.add_argument("--data", default=None,
                   help="Path to CSV with recency, frequency, monetary columns "
                        "(default: synthetic data)")
    p.add_argument("--B", type=int, default=50,
                   help="Resamples per perturbation type (use 100+ for final results)")
    p.add_argument("--out", default="output", help="Directory for result files")
    return p.parse_args()


# ---------------- 1. Data and preprocessing ----------------
def make_synthetic(n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    # (share, name, mean of [log recency, log frequency, log monetary])
    specs = [(0.35, "Champions", [2.0, 2.8, 6.5]),
             (0.30, "Regulars",  [3.5, 1.8, 4.8]),
             (0.20, "At-risk",   [4.8, 1.3, 4.4]),
             (0.15, "Lost/low",  [5.6, 0.5, 3.0])]
    parts = []
    for share, name, mu in specs:
        m = int(n * share)
        z = rng.normal(mu, 0.55, size=(m, 3))
        parts.append(pd.DataFrame(np.expm1(z), columns=FEATURES).assign(true_segment=name))
    return (pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=seed).reset_index(drop=True))


# ---------------- 2. Candidate models ----------------
def make_candidates():
    c = {}
    for k in K_RANGE:
        c[f"kmeans_k{k}"] = lambda k=k: KMeans(k, n_init=5, random_state=SEED)
        c[f"ward_k{k}"] = lambda k=k: AgglomerativeClustering(n_clusters=k, linkage="ward")
        c[f"gmm_k{k}"] = lambda k=k: GaussianMixture(k, random_state=SEED)
    for eps in DBSCAN_EPS:
        c[f"dbscan_eps{eps}"] = lambda e=eps: DBSCAN(eps=e, min_samples=DBSCAN_MIN_SAMPLES)
    return c


# ---------------- 3. Quality metrics ----------------
def quality(Z, labels):
    """DBSCAN noise (-1) is excluded; a model needs >= 2 clusters to be scored."""
    m = labels != -1
    if m.sum() < 10 or len(np.unique(labels[m])) < 2:
        return None
    return (silhouette_score(Z[m], labels[m]),
            calinski_harabasz_score(Z[m], labels[m]),
            davies_bouldin_score(Z[m], labels[m]))


# ---------------- 4. Stability under perturbation ----------------
def stability(make, ref_labels, mode, X, Xlog, B):
    rng = np.random.default_rng(SEED)      # same perturbations for every candidate
    n, out = len(X), []
    for _ in range(B):
        if mode == "subsample":
            idx = rng.choice(n, int(SUBSAMPLE_FRAC * n), replace=False)
            Z = StandardScaler().fit_transform(Xlog[idx])
        else:
            idx = np.arange(n)
            Z = X + rng.normal(0, NOISE_SD, X.shape)
        lab = make().fit_predict(Z)
        out.append(adjusted_rand_score(ref_labels[idx], lab))
    return np.array(out)


# ---------------- 7. Per-segment stability (Jaccard) ----------------
def per_cluster_jaccard(make, ref, X, Xlog, B):
    rng = np.random.default_rng(SEED)
    n = len(X)
    clusters = sorted(set(ref) - {-1})
    scores = {c: [] for c in clusters}
    for _ in range(B):
        idx = rng.choice(n, int(SUBSAMPLE_FRAC * n), replace=False)
        lab = make().fit_predict(StandardScaler().fit_transform(Xlog[idx]))
        ref_sub = ref[idx]
        for c in clusters:
            a = ref_sub == c
            best_j = 0.0
            for d in set(lab) - {-1}:
                b = lab == d
                best_j = max(best_j, (a & b).sum() / max((a | b).sum(), 1))
            scores[c].append(best_j)
    return pd.DataFrame({"segment": clusters,
                         "size": [(ref == c).sum() for c in clusters],
                         "jaccard_mean": [np.mean(scores[c]) for c in clusters],
                         "jaccard_sd": [np.std(scores[c]) for c in clusters]})


def main():
    args = parse_args()
    B = args.B
    os.makedirs(args.out, exist_ok=True)

    # ---- 1. Load data ----
    df = make_synthetic() if args.data is None else pd.read_csv(args.data)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"Input data is missing required column(s): {missing}")
    print("Data shape:", df.shape)
    print(df.head(), "\n")

    # Log-transform skewed features. Scaling is re-fit inside each resample (no leakage).
    Xlog = np.log1p(df[FEATURES].clip(lower=0).values)
    X = StandardScaler().fit_transform(Xlog)   # reference (full-data) representation

    # ---- 2. Candidates ----
    candidates = make_candidates()
    print(len(candidates), "candidate models")

    # ---- 3 & 4. Quality + stability for every candidate ----
    rows, ref_store, skipped = [], {}, []
    for name, make in candidates.items():
        ref = make().fit_predict(X)
        q = quality(X, ref)
        if q is None:
            skipped.append(name)
            continue
        ref_store[name] = ref
        s_sub = stability(make, ref, "subsample", X, Xlog, B)
        s_noi = stability(make, ref, "noise", X, Xlog, B)
        rows.append(dict(
            model=name, algo=name.split("_")[0],
            n_clusters=len(set(ref) - {-1}), noise_frac=(ref == -1).mean(),
            sil=q[0], ch=q[1], db=q[2],
            ari_sub=s_sub.mean(), ari_sub_sd=s_sub.std(),
            ari_noise=s_noi.mean(), ari_noise_sd=s_noi.std()))
        print(f"  scored {name}")

    res = pd.DataFrame(rows)
    print("\nSkipped (fewer than 2 clusters):", skipped)

    # ---- 5. Combined selection criterion ----
    minmax = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)

    res["Q"] = (minmax(res.sil) + minmax(res.ch) + (1 - minmax(res.db))) / 3
    res["Stab"] = (((res.ari_sub - res.ari_sub_sd) +
                    (res.ari_noise - res.ari_noise_sd)) / 2).clip(0, 1)
    res["score"] = 2 * res.Q * res.Stab / (res.Q + res.Stab + 1e-12)
    res = res.sort_values("score", ascending=False).reset_index(drop=True)

    best = res.score.max()
    eligible = res[res.score >= best - PARSIMONY_TOL]
    chosen = eligible.sort_values(["n_clusters", "score"], ascending=[True, False]).iloc[0]
    CHOSEN = chosen.model

    cols = ["model", "n_clusters", "sil", "ch", "db", "Q",
            "ari_sub", "ari_noise", "Stab", "score"]
    print("\nTop 10 by combined score:")
    print(res[cols].head(10).round(3).to_string(index=False))
    print(f"\nSelected model: {CHOSEN}  "
          f"(score={chosen.score:.3f}, Q={chosen.Q:.3f}, Stab={chosen.Stab:.3f})")

    res.round(4).to_csv(os.path.join(args.out, "model_scores.csv"), index=False)

    # Does stability change the answer?
    cmp = pd.DataFrame({
        "criterion": ["Quality only (Q)", "Stability only (Stab)",
                      "Combined (harmonic mean)", "Combined + parsimony"],
        "best model": [res.loc[res.Q.idxmax(), "model"], res.loc[res.Stab.idxmax(), "model"],
                       res.loc[res.score.idxmax(), "model"], CHOSEN]})
    print("\nWinner under each criterion:")
    print(cmp.to_string(index=False))
    cmp.to_csv(os.path.join(args.out, "criterion_comparison.csv"), index=False)

    # ---- 6. Visual comparison and weight sensitivity ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"kmeans": "tab:blue", "ward": "tab:green", "gmm": "tab:orange", "dbscan": "tab:red"}
    for a, g in res.groupby("algo"):
        ax[0].scatter(g.Q, g.Stab, c=colors[a], label=a, s=60, alpha=.8)
    for _, r in res.iterrows():
        ax[0].annotate(r.model, (r.Q, r.Stab), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    c = res[res.model == CHOSEN].iloc[0]
    ax[0].scatter(c.Q, c.Stab, s=260, facecolors="none", edgecolors="k",
                  linewidths=2, label="selected")
    ax[0].set_xlabel("Quality (Q)")
    ax[0].set_ylabel("Stability (Stab)")
    ax[0].set_title("Quality vs stability (top-right is best)")
    ax[0].legend()

    top = res.head(10).iloc[::-1]
    ax[1].barh(top.model, top.score, color=[colors[a] for a in top.algo])
    ax[1].set_title("Top 10 by combined score")
    ax[1].set_xlabel("score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "quality_vs_stability.png"), dpi=150)
    plt.close()

    # Sensitivity: weighted score w*Q + (1-w)*Stab
    sens = []
    for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
        s = w * res.Q + (1 - w) * res.Stab
        sens.append(dict(w_quality=w, best_model=res.loc[s.idxmax(), "model"],
                         score=round(s.max(), 3)))
    sens = pd.DataFrame(sens)
    print("\nWeight sensitivity (a robust choice wins across weights):")
    print(sens.to_string(index=False))
    sens.to_csv(os.path.join(args.out, "weight_sensitivity.csv"), index=False)

    # ---- 7. Per-segment stability ----
    ref = ref_store[CHOSEN]
    seg_stab = per_cluster_jaccard(candidates[CHOSEN], ref, X, Xlog, B)
    seg_stab["reliable"] = seg_stab.jaccard_mean >= JACCARD_THRESHOLD
    print("\nPer-segment stability (Jaccard):")
    print(seg_stab.round(3).to_string(index=False))
    seg_stab.round(4).to_csv(os.path.join(args.out, "segment_stability.csv"), index=False)

    # ---- 8. Segment profiles ----
    prof = (df[FEATURES].assign(segment=ref).query("segment != -1")
            .groupby("segment")[FEATURES].mean())
    prof["size"] = pd.Series(ref).value_counts()
    prof["share_%"] = (100 * prof["size"] / len(ref)).round(1)
    print("\nSegment profiles:")
    print(prof.round(1).to_string())
    prof.round(2).to_csv(os.path.join(args.out, "segment_profiles.csv"))

    Z2 = PCA(2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=ref, cmap="tab10", s=8)
    plt.title(f"Final segments: {CHOSEN} (PCA view)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "final_segments_pca.png"), dpi=150)
    plt.close()

    # Save customer-level assignments
    out_df = df.copy()
    out_df["segment"] = ref
    out_df.to_csv(os.path.join(args.out, "customers_with_segments.csv"), index=False)

    if "true_segment" in df.columns:
        print("\nSanity check vs known synthetic segments, ARI =",
              round(adjusted_rand_score(df.true_segment, ref), 3))

    print(f"\nDone. Results written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
