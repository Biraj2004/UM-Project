"""
Multi-Objective Optimization Engine for Nassau Candy
-----------------------------------------------------
This module computes Pareto-optimal factory reallocation policies by
balancing two key business priorities:
1. Delivery Speed / Distance Reduction (Operational Efficiency)
2. Gross Profit Margin Preservation (Financial Safety)

It generates ranked Top-N factory reallocation recommendations.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geo_utils import FACTORIES, PRODUCT_FACTORY_MAP, calculate_haversine_distance
from src.data_pipeline import load_raw_data


def calculate_optimization_score(
    distance_saved_pct,
    margin_pct,
    weight_speed=0.5,
    weight_profit=0.5,
    risk_penalty=0.0
):
    """
    Computes a normalized Multi-Objective Optimization Score (0 to 100).
    
    Parameters:
    - distance_saved_pct: Percentage of freight distance eliminated (0 to 100%)
    - margin_pct: Gross profit margin percentage (0 to 100%)
    - weight_speed: Importance of speed / distance reduction (0.0 to 1.0)
    - weight_profit: Importance of gross profit margin (0.0 to 1.0)
    - risk_penalty: Variance / capacity constraint penalty
    """
    # Normalize weights so they sum to 1.0
    total_w = weight_speed + weight_profit
    if total_w > 0:
        w_s = weight_speed / total_w
        w_p = weight_profit / total_w
    else:
        w_s, w_p = 0.5, 0.5

    score = (w_s * max(distance_saved_pct, 0)) + (w_p * max(margin_pct, 0)) - risk_penalty
    return round(max(score, 0.0), 2)


def generate_regional_recommendations(
    enriched_df=None,
    weight_speed=0.6,
    weight_profit=0.4
):
    """
    Analyzes all unique Product x Region order flows and generates
    ranked factory reallocation suggestions.
    """
    if enriched_df is None:
        file_path = "data/processed/nassau_candy_enriched.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}. Run src/data_pipeline.py first.")
        enriched_df = pd.read_csv(file_path)

    # Group by Product and Region
    grouped = enriched_df.groupby(['Product Name', 'Region', 'Current Factory']).agg(
        Order_Count=('Row ID', 'count'),
        Total_Units=('Units', 'sum'),
        Total_Sales=('Sales', 'sum'),
        Total_Cost=('Cost', 'sum'),
        Total_Gross_Profit=('Gross Profit', 'sum'),
        Avg_Current_Distance=('Transit Distance (Miles)', 'mean'),
        Avg_Optimal_Distance=('Minimum Distance (Miles)', 'mean'),
        Total_Potential_Miles_Saved=('Potential Distance Saved (Miles)', 'sum'),
        Avg_Margin_Pct=('Gross Margin %', 'mean'),
        Cust_Lat=('Customer Latitude', 'mean'),
        Cust_Lon=('Customer Longitude', 'mean')
    ).reset_index()

    recommendations = []

    for _, row in grouped.iterrows():
        product = row['Product Name']
        region = row['Region']
        curr_factory = row['Current Factory']
        order_count = int(row['Order_Count'])
        c_lat = row['Cust_Lat']
        c_lon = row['Cust_Lon']
        curr_dist = row['Avg_Current_Distance']
        margin = row['Avg_Margin_Pct']

        # Evaluate all 5 candidate factories for this product & region centroid
        candidate_scores = []
        for factory_name, f_info in FACTORIES.items():
            f_lat = f_info['latitude']
            f_lon = f_info['longitude']
            dist = calculate_haversine_distance(f_lat, f_lon, c_lat, c_lon)
            
            dist_saved = curr_dist - dist
            dist_saved_pct = (dist_saved / curr_dist * 100) if curr_dist > 0 else 0.0
            
            score = calculate_optimization_score(
                distance_saved_pct=dist_saved_pct,
                margin_pct=margin,
                weight_speed=weight_speed,
                weight_profit=weight_profit
            )

            candidate_scores.append({
                'Factory': factory_name,
                'Distance': dist,
                'Distance_Saved': dist_saved,
                'Distance_Saved_Pct': dist_saved_pct,
                'Score': score
            })

        # Find best candidate factory
        cand_df = pd.DataFrame(candidate_scores).sort_values(by='Score', ascending=False)
        best_candidate = cand_df.iloc[0]
        rec_factory = best_candidate['Factory']
        
        is_reassignment_recommended = (rec_factory != curr_factory and best_candidate['Distance_Saved'] > 100)
        
        expected_total_miles_saved = round(best_candidate['Distance_Saved'] * order_count, 0) if is_reassignment_recommended else 0.0

        recommendations.append({
            'Product Name': product,
            'Destination Region': region,
            'Current Factory': curr_factory,
            'Recommended Factory': rec_factory if is_reassignment_recommended else curr_factory,
            'Action': 'Reassign Factory' if is_reassignment_recommended else 'Keep Current',
            'Order Volume': order_count,
            'Current Distance (Miles)': round(curr_dist, 1),
            'New Distance (Miles)': round(best_candidate['Distance'], 1) if is_reassignment_recommended else round(curr_dist, 1),
            'Distance Reduction (%)': round(best_candidate['Distance_Saved_Pct'], 1) if is_reassignment_recommended else 0.0,
            'Total Freight Miles Saved': max(expected_total_miles_saved, 0),
            'Profit Margin (%)': round(margin, 1),
            'Optimization Score': best_candidate['Score']
        })

    rec_df = pd.DataFrame(recommendations)
    
    # Sort recommendations by highest freight mileage eliminated
    rec_df = rec_df.sort_values(by=['Action', 'Total Freight Miles Saved'], ascending=[True, False]).reset_index(drop=True)
    return rec_df


def run_optimization_pipeline():
    """
    Runs the recommendation engine and exports the top recommendations table.
    """
    print("Running Multi-Objective Optimization & Policy Recommendation Pipeline...")
    recs_df = generate_regional_recommendations(weight_speed=0.6, weight_profit=0.4)
    
    output_path = "data/processed/top_recommendations.csv"
    recs_df.to_csv(output_path, index=False)
    print(f"Top recommendations table saved to: {output_path}")

    # Display Top 10 Actionable Reassignments
    actionable = recs_df[recs_df['Action'] == 'Reassign Factory']
    print(f"\n--- Top {min(len(actionable), 8)} Actionable Factory Reassignments ---")
    print(actionable[['Product Name', 'Destination Region', 'Current Factory', 'Recommended Factory', 'Distance Reduction (%)', 'Total Freight Miles Saved']].head(8).to_string(index=False))

    return recs_df


if __name__ == "__main__":
    run_optimization_pipeline()
