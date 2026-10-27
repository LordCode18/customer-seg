import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

warnings.filterwarnings("ignore")
FEATURES = ["recency", "frequency", "monetary"]
SEED = 42

st.set_page_config(page_title="Customer Segmentation Stability", page_icon="📊", layout="wide")
st.title("Customer Segmentation with Cluster Stability")
st.write("Compare clustering algorithms using both cluster quality and stability under repeated dataset perturbations.")

with st.sidebar:
    st.header("Settings")
    B = st.slider("Resamples per perturbation", min_value=5, max_value=50, value=10, step=5,
                  help="Higher values give more reliable stability estimates but take longer.")
    k_min = st.slider("Minimum number of clusters (k)", 2, 4, 2)
    k_max = st.slider("Maximum number of clusters (k)", max(3, k_min + 1), 10, max(5, k_min + 1))
    subsample_frac = st.slider("Subsample fraction", 0.5, 0.95, 0.8, 0.05)
    noise_sd = st.slider("Noise level", 0.05, 0.5, 0.15, 0.05)
    parsimony_tol = st.slider("Parsimony tolerance", 0.0, 0.1, 0.02, 0.01)
    jaccard_threshold = st.slider("Reliable-segment Jaccard threshold", 0.5, 0.95, 0.75, 0.05)

uploaded = st.file_uploader("Upload customer data (CSV or Excel). Accepts RFM columns or raw Online Retail II transaction data.", type=["csv", "xlsx", "xls"])

def make_synthetic(n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    specs = [(0.35, "Champions", [2.0, 2.8, 6.5]),
             (0.30, "Regulars", [3.5, 1.8, 4.8]),
             (0.20, "At-risk", [4.8, 1.3, 4.4]),
             (0.15, "Lost/low", [5.6, 0.5, 3.0])]
    parts = []
    for share, name, mu in specs:
        m = int(n * share)
        z = rng.normal(mu, 0.55, size=(m, 3))
        parts.append(pd.DataFrame(np.expm1(z), columns=FEATURES).assign(true_segment=name))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def read_data(file_bytes, file_name):
    import io
    bio = io.BytesIO(file_bytes)
    if file_name.lower().endswith('.csv'):
        return pd.read_csv(bio)
    return pd.read_excel(bio)

if uploaded is not None:
    try:
        df = read_data(uploaded.getvalue(), uploaded.name)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        st.stop()
    st.caption(f"Loaded uploaded dataset: {uploaded.name}")

    # Accept either customer-level RFM data or raw Online Retail II transactions.
    normalized = {str(c).strip().lower(): c for c in df.columns}
    rfm_keys = {"recency", "frequency", "monetary"}
    if not rfm_keys.issubset(normalized):
        customer_col = next((normalized[k] for k in ("customer id", "customer_id", "customerid") if k in normalized), None)
        date_col = next((normalized[k] for k in ("invoicedate", "invoice date", "invoice_date") if k in normalized), None)
        invoice_col = next((normalized[k] for k in ("invoiceno", "invoice no", "invoice_no") if k in normalized), None)
        quantity_col = normalized.get("quantity")
        price_col = next((normalized[k] for k in ("price", "unitprice", "unit price") if k in normalized), None)
        if all(x is not None for x in (customer_col, date_col, invoice_col, quantity_col, price_col)):
            tx = df[[customer_col, date_col, invoice_col, quantity_col, price_col]].copy()
            tx.columns = ["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "Price"]
            tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"], errors="coerce")
            tx["Quantity"] = pd.to_numeric(tx["Quantity"], errors="coerce")
            tx["Price"] = pd.to_numeric(tx["Price"], errors="coerce")
            tx = tx.dropna(subset=["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "Price"])
            tx = tx[(tx["Quantity"] > 0) & (tx["Price"] > 0) & (~tx["InvoiceNo"].astype(str).str.startswith("C"))]
            tx["Sales"] = tx["Quantity"] * tx["Price"]
            if tx.empty:
                st.error("No valid positive-quantity transaction rows found after cleaning.")
                st.stop()
            snapshot_date = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
            df = tx.groupby("CustomerID").agg(
                last_purchase=("InvoiceDate", "max"),
                frequency=("InvoiceNo", "nunique"),
                monetary=("Sales", "sum"),
            ).reset_index()
            df["recency"] = (snapshot_date - df["last_purchase"].dt.normalize()).dt.days
            df = df[FEATURES]
            st.info("Converted raw retail transactions into customer-level RFM. Cancellation invoices and non-positive quantity/price rows were excluded.")
        else:
            st.error("Upload RFM columns (recency, frequency, monetary), or raw Online Retail II columns for Customer ID, InvoiceDate, InvoiceNo, Quantity, and Price.")
            st.stop()
else:
    df = make_synthetic()
    st.info("Using demonstration synthetic data. Upload your own data to analyze real customers.")

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=FEATURES).copy()
df[FEATURES] = df[FEATURES].clip(lower=0)
if len(df) < 20:
    st.error("Please provide at least 20 rows with valid numeric recency, frequency, and monetary values.")
    st.stop()

st.subheader("Data preview")
st.write(f"**{len(df):,} customers** · **{len(df.columns)} columns**")
st.dataframe(df.head(10), use_container_width=True)

if st.button("Run segmentation analysis", type="primary"):
    with st.spinner("Testing clustering models and stability. This can take a few minutes..."):
        Xlog = np.log1p(df[FEATURES].values)
        X = StandardScaler().fit_transform(Xlog)
        n = len(X)
        k_range = range(k_min, k_max + 1)
        candidates = {}
        for k in k_range:
            candidates[f"kmeans_k{k}"] = lambda k=k: KMeans(n_clusters=k, n_init=5, random_state=SEED)
            candidates[f"ward_k{k}"] = lambda k=k: AgglomerativeClustering(n_clusters=k, linkage="ward")
            candidates[f"gmm_k{k}"] = lambda k=k: GaussianMixture(n_components=k, random_state=SEED)
        for eps in [0.3, 0.5, 0.7]:
            candidates[f"dbscan_eps{eps}"] = lambda e=eps: DBSCAN(eps=e, min_samples=20)

        def quality(Z, labels):
            mask = labels != -1
            if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
                return None
            return (silhouette_score(Z[mask], labels[mask]),
                    calinski_harabasz_score(Z[mask], labels[mask]),
                    davies_bouldin_score(Z[mask], labels[mask]))

        def stability(make, ref_labels, mode):
            rng = np.random.default_rng(SEED)
            scores = []
            for _ in range(B):
                if mode == "subsample":
                    idx = rng.choice(n, max(10, int(subsample_frac * n)), replace=False)
                    Z = StandardScaler().fit_transform(Xlog[idx])
                else:
                    idx = np.arange(n)
                    Z = X + rng.normal(0, noise_sd, X.shape)
                labels = make().fit_predict(Z)
                scores.append(adjusted_rand_score(ref_labels[idx], labels))
            return np.asarray(scores)

        rows, ref_store, skipped = [], {}, []
        progress = st.progress(0, text="Evaluating candidate models...")
        for i, (name, make) in enumerate(candidates.items()):
            ref = make().fit_predict(X)
            q = quality(X, ref)
            if q is None:
                skipped.append(name)
                progress.progress((i + 1) / len(candidates), text=f"Evaluating {name}...")
                continue
            ref_store[name] = ref
            s_sub = stability(make, ref, "subsample")
            s_noise = stability(make, ref, "noise")
            rows.append(dict(model=name, algo=name.split("_")[0],
                             n_clusters=len(set(ref) - {-1}), noise_frac=(ref == -1).mean(),
                             sil=q[0], ch=q[1], db=q[2],
                             ari_sub=s_sub.mean(), ari_sub_sd=s_sub.std(),
                             ari_noise=s_noise.mean(), ari_noise_sd=s_noise.std()))
            progress.progress((i + 1) / len(candidates), text=f"Evaluating {name}...")
        progress.empty()
        if not rows:
            st.error("No candidate model produced at least two usable clusters. Try different data or settings.")
            st.stop()
        res = pd.DataFrame(rows)
        minmax = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
        res["Q"] = (minmax(res.sil) + minmax(res.ch) + (1 - minmax(res.db))) / 3
        res["Stab"] = (((res.ari_sub - res.ari_sub_sd) + (res.ari_noise - res.ari_noise_sd)) / 2).clip(0, 1)
        res["score"] = 2 * res.Q * res.Stab / (res.Q + res.Stab + 1e-12)
        res = res.sort_values("score", ascending=False).reset_index(drop=True)
        best = res.score.max()
        eligible = res[res.score >= best - parsimony_tol]
        chosen = eligible.sort_values(["n_clusters", "score"], ascending=[True, False]).iloc[0]
        chosen_name = chosen.model
        ref = ref_store[chosen_name]

        st.success(f"Selected model: {chosen_name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Combined score", f"{chosen.score:.3f}")
        m2.metric("Quality (Q)", f"{chosen.Q:.3f}")
        m3.metric("Stability", f"{chosen.Stab:.3f}")
        m4.metric("Clusters", int(chosen.n_clusters))

        st.subheader("Model comparison")
        cols = ["model", "n_clusters", "sil", "ch", "db", "Q", "ari_sub", "ari_noise", "Stab", "score"]
        st.dataframe(res[cols].round(3), use_container_width=True)
        cmp = pd.DataFrame({
            "criterion": ["Quality only (Q)", "Stability only (Stab)", "Combined score", "Combined + parsimony"],
            "best model": [res.loc[res.Q.idxmax(), "model"], res.loc[res.Stab.idxmax(), "model"],
                           res.loc[res.score.idxmax(), "model"], chosen_name]})
        st.subheader("Does stability change the answer?")
        st.dataframe(cmp, hide_index=True, use_container_width=True)

        st.subheader("Quality versus stability")
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"kmeans": "tab:blue", "ward": "tab:green", "gmm": "tab:orange", "dbscan": "tab:red"}
        for algo, group in res.groupby("algo"):
            ax.scatter(group.Q, group.Stab, c=colors.get(algo, "gray"), label=algo, s=60, alpha=.8)
        for _, row in res.iterrows():
            ax.annotate(row.model, (row.Q, row.Stab), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.scatter(chosen.Q, chosen.Stab, s=250, facecolors="none", edgecolors="black", linewidths=2, label="selected")
        ax.set_xlabel("Quality (Q)"); ax.set_ylabel("Stability (Stab)"); ax.legend(); ax.grid(alpha=.2)
        st.pyplot(fig); plt.close(fig)

        sens = []
        for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
            score = w * res.Q + (1 - w) * res.Stab
            sens.append(dict(quality_weight=w, best_model=res.loc[score.idxmax(), "model"], score=round(score.max(), 3)))
        st.subheader("Weight sensitivity")
        st.dataframe(pd.DataFrame(sens), hide_index=True, use_container_width=True)

        rng = np.random.default_rng(SEED)
        clusters = sorted(set(ref) - {-1})
        scores = {c: [] for c in clusters}
        for _ in range(B):
            idx = rng.choice(n, max(10, int(subsample_frac * n)), replace=False)
            labels = candidates[chosen_name]().fit_predict(StandardScaler().fit_transform(Xlog[idx]))
            ref_sub = ref[idx]
            for c in clusters:
                a = ref_sub == c
                best_j = 0.0
                for d in set(labels) - {-1}:
                    b = labels == d
                    best_j = max(best_j, (a & b).sum() / max((a | b).sum(), 1))
                scores[c].append(best_j)
        seg_stab = pd.DataFrame({"segment": clusters,
                                 "size": [(ref == c).sum() for c in clusters],
                                 "jaccard_mean": [np.mean(scores[c]) for c in clusters],
                                 "jaccard_sd": [np.std(scores[c]) for c in clusters]})
        seg_stab["reliable"] = seg_stab.jaccard_mean >= jaccard_threshold
        st.subheader("Per-segment stability")
        st.caption(f"Segments with mean Jaccard similarity ≥ {jaccard_threshold:.2f} are marked reliable.")
        st.dataframe(seg_stab.round(3), hide_index=True, use_container_width=True)

        prof = df[FEATURES].assign(segment=ref).query("segment != -1").groupby("segment")[FEATURES].mean()
        prof["size"] = pd.Series(ref).value_counts()
        prof["share_%"] = (100 * prof["size"] / len(ref)).round(1)
        st.subheader("Segment profiles")
        st.dataframe(prof.round(2), use_container_width=True)

        st.subheader("PCA visualization of selected segments")
        Z2 = PCA(n_components=2, random_state=SEED).fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(Z2[:, 0], Z2[:, 1], c=ref, cmap="tab10", s=10, alpha=.75)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"Final segments: {chosen_name}")
        st.pyplot(fig); plt.close(fig)

        if "true_segment" in df.columns:
            st.caption(f"Synthetic-data sanity check (ARI against known labels): {adjusted_rand_score(df.true_segment, ref):.3f}")
        if skipped:
            st.caption("Skipped models with fewer than two usable clusters: " + ", ".join(skipped))
        st.download_button("Download model comparison (CSV)", res.to_csv(index=False).encode("utf-8"),
                           file_name="segmentation_model_comparison.csv", mime="text/csv")
else:
    st.caption("Adjust settings in the sidebar, upload data if needed, then click **Run segmentation analysis**.")
