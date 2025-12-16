import os
import argparse
import warnings
import pandas as pd
import numpy as np
import datetime as dt
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# import joblib
# import matplotlib
# matplotlib.use('Agg') # Use non-interactive backend
# import matplotlib.pyplot as plt

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
# ============================
# CAP OUTLIERS + LOG TRANSFORM
# ============================
def cap_and_log_transform(rfm):
    """Cap extreme values (99th percentile), log-transform, then scale manually."""
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

    # Manual Standard Scaling to avoid sklearn dependency
    # (x - mean) / std
    rfm_scaled_dict = {}
    for col in ["R_log", "F_log", "M_log"]:
        mean = rfm_log[col].mean()
        std = rfm_log[col].std(ddof=0) # Population std or sample std? sklearn uses biased=False by default? No, StandardScaler uses with_mean=True, with_std=True. It uses biased estimator (ddof=0) for std? 
        # sklearn StandardScaler uses np.std(x, axis=0) which defaults to ddof=0. 
        if std == 0:
            rfm_scaled_dict[f"{col}_sc"] = rfm_log[col] - mean
        else:
            rfm_scaled_dict[f"{col}_sc"] = (rfm_log[col] - mean) / std

    rfm_scaled_df = pd.DataFrame(rfm_scaled_dict, index=rfm_proc.index)

    return rfm_proc, rfm_log, rfm_scaled_df, None


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
# ============================
# ARGPARSE and MAIN removed to avoid dependencies
# ============================
