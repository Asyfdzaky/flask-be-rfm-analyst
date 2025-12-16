import os
import argparse
import warnings
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================
# LOAD DATA
# ============================
def load_data(path):
    """Load dataset from Excel or CSV."""
    if path.lower().endswith(".xlsx") or path.lower().endswith(".xls"):
        df = pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Provide .xlsx, .xls or .csv")
    return df


# ============================
# BASIC CLEANING
# ============================
def basic_cleaning(df):
    """Drop missing customer IDs and remove non-positive quantity/unitprice."""
    df = df.copy()
    # Convert to numeric, coercing non-numeric values to NaN
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors='coerce')

    # Drop rows where Quantity or UnitPrice is NaN (failed conversion)
    df.dropna(subset=["Quantity", "UnitPrice"], inplace=True)

    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # ensure InvoiceDate is datetime
    # Handle odd formats like "01/12/2010 08.45" where dot is used for time
    if df["InvoiceDate"].dtype == 'object':
         df["InvoiceDate"] = df["InvoiceDate"].astype(str).str.replace(".", ":", regex=False)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors='coerce')
    
    # Drop rows where InvoiceDate failed to parse
    df.dropna(subset=["InvoiceDate"], inplace=True)

    # compute Amount
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    return df


# ============================
# COMPUTE RFM
# ============================
def compute_rfm(df, reference_date=None):
    """Compute RFM table per CustomerID."""
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Amount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    return rfm


# ============================
# CAP OUTLIERS + LOG TRANSFORM
# ============================
def cap_and_log_transform(rfm):
    """Cap extreme values (99th percentile), log-transform, then scale."""
    rfm_proc = rfm.copy()

    Q99_F = rfm_proc["Frequency"].quantile(0.99)
    Q99_M = rfm_proc["Monetary"].quantile(0.99)

    rfm_proc["Frequency_Capped"] = np.where(rfm_proc["Frequency"] > Q99_F, Q99_F, rfm_proc["Frequency"])
    rfm_proc["Monetary_Capped"] = np.where(rfm_proc["Monetary"] > Q99_M, Q99_M, rfm_proc["Monetary"])
    rfm_proc["Recency_Capped"] = rfm_proc["Recency"]

    rfm_log = pd.DataFrame({
        "R_log": np.log(rfm_proc["Recency_Capped"] + 1),
        "F_log": np.log(rfm_proc["Frequency_Capped"] + 1),
        "M_log": np.log(rfm_proc["Monetary_Capped"] + 1)
    })

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    rfm_scaled_df = pd.DataFrame(
        rfm_scaled,
        columns=["R_log_sc", "F_log_sc", "M_log_sc"],
        index=rfm_proc.index
    )

    return rfm_proc, rfm_log, rfm_scaled_df, scaler


# ============================
# EVALUATE BEST K
# ============================
def evaluate_k_options(rfm_scaled, k_min=2, k_max=10, output_dir="./output"):
    """Compute WCSS & Silhouette scores for K range and save plots."""
    wcss = []
    sil_scores = []
    K_range = range(k_min, k_max + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(rfm_scaled, km.labels_))

    # save silhouette plot
    plt.figure(figsize=(8, 5))
    plt.bar(list(K_range), sil_scores)
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score per K")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "silhouette_per_k.png"))
    plt.close()

    # elbow plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(K_range), wcss, marker="o", linestyle="--")
    plt.xlabel("K")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "elbow_wcss.png"))
    plt.close()

    best_k = K_range[int(np.argmax(sil_scores))]
    return {"K_range": list(K_range), "wcss": wcss, "silhouette": sil_scores, "best_k_by_silhouette": best_k}


# ============================
# FINAL MODELS + SAVE RESULTS
# ============================
def fit_and_save_models(rfm_orig, rfm_scaled_df, k_final=5, output_dir="./output"):
    """Fit KMeans + alternative clustering models and save results."""
    os.makedirs(output_dir, exist_ok=True)

    # KMeans final model
    kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
    labels_km = kmeans_final.fit_predict(rfm_scaled_df)
    rfm_orig["Cluster"] = labels_km

    # Save the model
    joblib.dump(kmeans_final, os.path.join(output_dir, "rfm_kmeans.model"))

    # MiniBatch
    minibatch = MiniBatchKMeans(n_clusters=k_final, random_state=42, n_init=10)
    rfm_orig["MiniBatch_Cluster"] = minibatch.fit_predict(rfm_scaled_df)

    # Hierarchical
    hier = AgglomerativeClustering(n_clusters=k_final)
    rfm_orig["Hierarchical_Cluster"] = hier.fit_predict(rfm_scaled_df)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    rfm_orig["DBSCAN_Cluster"] = dbscan.fit_predict(rfm_scaled_df)

    # Metrics
    metrics = {
        "KMeans": {
            "Silhouette": silhouette_score(rfm_scaled_df, rfm_orig["Cluster"]),
            "Davies-Bouldin": davies_bouldin_score(rfm_scaled_df, rfm_orig["Cluster"]),
            "Calinski-Harabasz": calinski_harabasz_score(rfm_scaled_df, rfm_orig["Cluster"])
        }
    }

    # Save clustered CSV
    rfm_orig.to_csv(os.path.join(output_dir, "rfm_clustered.csv"), index=False)

    return rfm_orig, metrics


# ============================
# PLOTS & PROFILES
# ============================
def create_plots_and_profiles(rfm, output_dir="./output"):
    """Create RFM boxplots, scatter, pie chart, and cluster profile."""
    os.makedirs(output_dir, exist_ok=True)

    # RFM boxplots
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(["Recency", "Frequency", "Monetary"], 1):
        plt.subplot(1, 3, i)
        plt.boxplot(rfm[col].dropna())
        plt.title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rfm_boxplots.png"))
    plt.close()

    # Scatter plot
    plt.figure(figsize=(10, 8))
    for c in sorted(rfm["Cluster"].unique()):
        subset = rfm[rfm["Cluster"] == c]
        plt.scatter(subset["Frequency"], subset["Monetary"], label=f"Cluster {c}", s=20)
    plt.xlabel("Frequency")
    plt.ylabel("Monetary")
    plt.title("Frequency vs Monetary by Cluster")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, "rfm_scatter.png"))
    plt.close()

    # Pie chart
    profile = rfm.groupby("Cluster")["CustomerID"].count().rename("Count")
    profile_ratio = profile / profile.sum() * 100

    plt.figure(figsize=(8, 8))
    plt.pie(profile_ratio, labels=[f"C{i}" for i in profile_ratio.index], autopct="%1.1f%%")
    plt.title("Customer Distribution per Cluster")
    plt.savefig(os.path.join(output_dir, "cluster_pie.png"))
    plt.close()

    # Export cluster profile
    cluster_profile = rfm.groupby("Cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        Count=("CustomerID", "count")
    )
    cluster_profile["Percentage"] = cluster_profile["Count"] / cluster_profile["Count"].sum() * 100

    cluster_profile.to_csv(os.path.join(output_dir, "cluster_profile.csv"))


# ============================
# AUTOMATIC SEGMENT LABELING
# ============================
def label_segments_auto(cluster_profile):
    """Manual cluster labeling sesuai definisi Google Docs."""
    
    mapping = {
	0: "Pelanggan yang Tidak Aktif/Hilang",
        1: "Champion/Pelanggan Setia",
        2: "Potensi Setia/Pelanggan yang Membutuhkan Perhatian",
        3: "Pelanggan Baru/Berisiko",
        4: "Pembelanja Besar/Pelanggan Baru Bernilai Tinggi"
    }
    
    cluster_profile = cluster_profile.copy()
    cluster_profile["Segment"] = cluster_profile.index.map(mapping)
    return cluster_profile

# ============================
# ARGPARSE
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="RFM pipeline for Online Retail (Mode 3)")
    parser.add_argument("--input", required=True, help="Input Excel/CSV path")
    parser.add_argument("--output_dir", default="./output", help="Directory to save results")
    parser.add_argument("--k", type=int, default=5, help="Final K for KMeans (default 5)")
    parser.add_argument("--kmin", type=int, default=2, help="Min K to evaluate")
    parser.add_argument("--kmax", type=int, default=10, help="Max K to evaluate")
    return parser.parse_args()


# ============================
# MAIN
# ============================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(args.input)
    print(f"Original rows: {len(df):,}")

    print("Cleaning...")
    df = basic_cleaning(df)
    print(f"After cleaning rows: {len(df):,}")

    print("Computing RFM...")
    rfm = compute_rfm(df)

    print("Transforming data...")
    rfm_proc, rfm_log, rfm_scaled_df, scaler = cap_and_log_transform(rfm)

    print("Evaluating K...")
    eval_res = evaluate_k_options(rfm_scaled_df.values, args.kmin, args.kmax, args.output_dir)
    print("Suggested K by silhouette:", eval_res["best_k_by_silhouette"])

    print(f"Fitting final models (k={args.k})...")
    rfm_clustered, metrics = fit_and_save_models(rfm, rfm_scaled_df, args.k, args.output_dir)
    print("Metrics:", metrics)

    print("Creating plots...")
    create_plots_and_profiles(rfm_clustered, args.output_dir)

    print("Labeling segments...")
    profile = pd.read_csv(os.path.join(args.output_dir, "cluster_profile.csv"), index_col=0)
    labeled = label_segments_auto(profile)
    labeled.to_csv(os.path.join(args.output_dir, "cluster_profile_labeled.csv"))

    print("Pipeline complete. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
