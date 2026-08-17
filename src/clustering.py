"""
Route & Bottleneck Clustering Engine for Nassau Candy
------------------------------------------------------
This module uses Unsupervised Machine Learning (K-Means Clustering)
to automatically analyze shipping routes and group them into:
1. Local / High-Efficiency Routes
2. Balanced Regional Routes
3. Severe High-Latency Bottleneck Corridors (Priority for Factory Reallocation)
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_processed_data(file_path="data/processed/nassau_candy_enriched.csv"):
    """
    Loads the enriched dataset from data/processed/.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed data not found at {file_path}. Please run src/data_pipeline.py first."
        )
    return pd.read_csv(file_path)


def perform_route_clustering(df, n_clusters=3, random_state=42):
    """
    Performs K-Means clustering on route performance features.
    
    Features used:
    - Transit Distance (Miles)
    - Lead Time (Days)
    - Units
    - Cost
    
    Returns:
    - Dataframe with assigned cluster labels and readable cluster descriptions.
    - Cluster summary statistics table.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    print(f"Running K-Means Route Clustering with k={n_clusters} clusters...")
    df = df.copy()

    feature_cols = ['Transit Distance (Miles)', 'Lead Time (Days)', 'Units', 'Cost']
    X = df[feature_cols].copy()

    # Step 1: Standardize features (mean=0, std=1) so distance and days have equal weight
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # Step 3: Rank clusters by average transit distance to give meaningful labels
    cluster_dist_rank = (
        df.groupby('Cluster_ID')['Transit Distance (Miles)']
        .mean()
        .sort_values()
        .index.tolist()
    )

    cluster_name_map = {
        cluster_dist_rank[0]: "Low Distance / Fast Delivery",
        cluster_dist_rank[1]: "Moderate Distance / Standard Route",
        cluster_dist_rank[2]: "High Latency / Bottleneck Route"
    }

    df['Cluster_Label'] = df['Cluster_ID'].map(cluster_name_map)

    # Step 4: Generate summary profile for each cluster
    summary = df.groupby('Cluster_Label').agg(
        Order_Count=('Row ID', 'count'),
        Avg_Distance_Miles=('Transit Distance (Miles)', 'mean'),
        Avg_Lead_Time_Days=('Lead Time (Days)', 'mean'),
        Avg_Cost=('Cost', 'mean'),
        Avg_Gross_Profit=('Gross Profit', 'mean'),
        Closer_Factory_Available=('Is Closer Factory Available', 'mean')
    ).reset_index()

    summary['Order_Share_%'] = (summary['Order_Count'] / len(df) * 100).round(1)
    summary['Closer_Factory_Available_%'] = (summary['Closer_Factory_Available'] * 100).round(1)
    summary = summary.drop(columns=['Closer_Factory_Available'])

    print("\n--- Route Clustering Summary ---")
    print(summary.to_string(index=False))

    return df, summary, kmeans, scaler


def get_top_bottleneck_routes(df, top_n=10):
    """
    Identifies the specific Factory -> State routes experiencing the worst
    distance and lead time bottlenecks.
    """
    route_agg = df.groupby(['Current Factory', 'State/Province', 'Region']).agg(
        Order_Count=('Row ID', 'count'),
        Avg_Distance_Miles=('Transit Distance (Miles)', 'mean'),
        Avg_Optimal_Distance=('Minimum Distance (Miles)', 'mean'),
        Avg_Distance_Saved=('Potential Distance Saved (Miles)', 'mean'),
        Total_Miles_Saved=('Potential Distance Saved (Miles)', 'sum'),
        Avg_Lead_Time=('Lead Time (Days)', 'mean')
    ).reset_index()

    # Filter to routes with high potential distance savings
    bottlenecks = (
        route_agg.sort_values(by='Total_Miles_Saved', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return bottlenecks


def run_clustering_pipeline():
    """
    Loads enriched data, performs clustering, and saves the clustered dataset.
    """
    df = load_processed_data()
    clustered_df, summary, model, scaler = perform_route_clustering(df)
    
    # Save clustered data
    output_path = "data/processed/nassau_candy_clustered.csv"
    clustered_df.to_csv(output_path, index=False)
    print(f"\nClustered dataset saved to: {output_path}")

    # Top bottleneck routes
    print("\n--- Top 5 Most Congested / Wasteful Routes ---")
    top_routes = get_top_bottleneck_routes(clustered_df, top_n=5)
    print(top_routes[['Current Factory', 'State/Province', 'Order_Count', 'Avg_Distance_Miles', 'Total_Miles_Saved']].to_string(index=False))

    return clustered_df, summary


if __name__ == "__main__":
    run_clustering_pipeline()
