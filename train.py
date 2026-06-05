"""
K-Means Clustering Model Training Script
Performs preprocessing, dynamic feature scaling, and trains a K-Means algorithm on the Mall Customers dataset.
Saves the trained model and scaler as .pkl files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    print("=" * 60)
    print("Initializing Model Training Pipeline...")
    print("=" * 60)

    # 1. Load the dataset
    data_path = os.path.join("data", "datasets.csv")
    if not os.path.exists(data_path):
        # Fallback to absolute paths if running dynamically
        data_path = os.path.join(os.path.dirname(__file__), "data", "datasets.csv")
    
    print(f"Reading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Successfully loaded dataset. Shape: {df.shape}")
    print("\nDataset columns available:")
    for col in df.columns:
        print(f" - {col}")

    # 2. Select Features for Clustering
    # Setting target columns for customer segmentation
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    print(f"\nSelected features for spatial segmentation: {features}")
    X = df[features].values

    # 3. Apply Feature Scaling
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features successfully normalized (mean=0, variance=1).")

    # 4. Initialize and Train K-Means
    k_clusters = 5
    print(f"\nTraining K-Means algorithm (k={k_clusters}, init='k-means++', random_state=42)...")
    kmeans = KMeans(n_clusters=k_clusters, init='k-means++', max_iter=300, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    
    # 5. Evaluate Clustering Model
    inertia = kmeans.inertia_
    labels = kmeans.labels_
    print(f"Training completed. Final Model Inertia (WCSS): {inertia:.4f}")
    
    # Show cluster counts
    unique_clusters, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f" - Cluster #{cluster_id}: {count} customers ({count / len(labels) * 100:.1f}%)")

    # 6. Save Artifacts to models/
    os.makedirs("models", exist_ok=True)
    model_out_path = os.path.join("models", "kmeans_model.pkl")
    scaler_out_path = os.path.join("models", "scaler.pkl")

    print(f"\nSaving model artifacts to disk...")
    joblib.dump(kmeans, model_out_path)
    joblib.dump(scaler, scaler_out_path)
    
    print(f"✓ Saved KMeans model to: {model_out_path}")
    print(f"✓ Saved MinMaxScaler/Scaler to: {scaler_out_path}")
    print("\nTraining workflow finalized successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
