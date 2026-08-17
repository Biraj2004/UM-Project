"""
Data Pipeline for Nassau Candy Factory Reallocation Project
------------------------------------------------------------
This module loads the raw transactional data, cleans it, calculates
lead times, computes geographical transit distances from all 5 factories,
and exports an enriched dataset for modeling and analysis.
"""

import os
import sys
import pandas as pd

# Add project root to sys.path so modules can be run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geo_utils import (
    FACTORIES,
    PRODUCT_FACTORY_MAP,
    get_customer_coordinates,
    calculate_haversine_distance,
    get_distance_to_factory
)


def load_raw_data(file_path="docs/Dataset - Nassau Candy Distributor.csv"):
    """
    Loads raw CSV data from the docs directory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    print(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def enrich_dataset(df):
    """
    Cleans and enriches raw transactional records with:
    1. Calculated Lead Time (in days).
    2. Assigned manufacturing factory.
    3. Customer destination GPS coordinates.
    4. Distance (in miles) from the assigned factory.
    5. Distances from all 5 candidate factories for comparison.
    6. Closest factory and distance savings.
    7. Unit economics (Unit Price, Unit Cost, Gross Margin %).
    8. Temporal order attributes (Month, Year, Day of Week).
    """
    df = df.copy()
    print("Enriching dataset with lead times and geospatial features...")

    # Step 1: Parse dates (format is DD-MM-YYYY)
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')

    # Step 2: Calculate Lead Time in days
    df['Lead Time (Days)'] = (df['Ship Date'] - df['Order Date']).dt.days

    # Step 3: Map each product to its default baseline factory
    df['Current Factory'] = df['Product Name'].map(PRODUCT_FACTORY_MAP)

    # Step 4: Add Customer GPS Coordinates (Latitude, Longitude)
    coords = [
        get_customer_coordinates(city, state)
        for city, state in zip(df['City'], df['State/Province'])
    ]
    df['Customer Latitude'] = [c[0] for c in coords]
    df['Customer Longitude'] = [c[1] for c in coords]

    # Step 5: Add Factory GPS Coordinates for the currently assigned factory
    df['Factory Latitude'] = df['Current Factory'].map(
        lambda f: FACTORIES[f]['latitude'] if f in FACTORIES else None
    )
    df['Factory Longitude'] = df['Current Factory'].map(
        lambda f: FACTORIES[f]['longitude'] if f in FACTORIES else None
    )

    # Step 6: Calculate current transit distance in miles
    current_distances = []
    for _, row in df.iterrows():
        f_lat = row['Factory Latitude']
        f_lon = row['Factory Longitude']
        c_lat = row['Customer Latitude']
        c_lon = row['Customer Longitude']
        dist = calculate_haversine_distance(f_lat, f_lon, c_lat, c_lon)
        current_distances.append(dist)
    df['Transit Distance (Miles)'] = current_distances

    # Step 7: Calculate distance to each of the 5 candidate factories
    for factory_name in FACTORIES.keys():
        clean_name = factory_name.replace(' ', '_').replace("'", "")
        col_name = f"Dist_{clean_name}"
        df[col_name] = [
            get_distance_to_factory(factory_name, lat, lon)
            for lat, lon in zip(df['Customer Latitude'], df['Customer Longitude'])
        ]

    # Step 8: Identify the geographically closest factory and potential distance savings
    factory_list = list(FACTORIES.keys())
    distance_cols = [
        f"Dist_{f.replace(' ', '_').replace(chr(39), '')}" for f in factory_list
    ]
    
    closest_factories = []
    min_distances = []

    for _, row in df.iterrows():
        dists = [row[col] for col in distance_cols]
        min_dist = min(dists)
        best_factory = factory_list[dists.index(min_dist)]
        closest_factories.append(best_factory)
        min_distances.append(min_dist)

    df['Closest Factory'] = closest_factories
    df['Minimum Distance (Miles)'] = min_distances
    df['Potential Distance Saved (Miles)'] = (
        df['Transit Distance (Miles)'] - df['Minimum Distance (Miles)']
    )
    df['Is Closer Factory Available'] = (
        df['Current Factory'] != df['Closest Factory']
    )

    # Step 9: Unit Economics & Financial Features
    df['Unit Price'] = (df['Sales'] / df['Units']).round(2)
    df['Unit Cost'] = (df['Cost'] / df['Units']).round(2)
    df['Gross Margin %'] = ((df['Gross Profit'] / df['Sales']) * 100).round(2)

    # Step 10: Temporal Calendar Features
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Month Name'] = df['Order Date'].dt.month_name()
    df['Order Day of Week'] = df['Order Date'].dt.day_name()
    df['Order Quarter'] = df['Order Date'].dt.quarter

    print("Data enrichment complete.")
    return df


def save_processed_data(df, output_path="data/processed/nassau_candy_enriched.csv"):
    """
    Saves the enriched dataframe to a CSV file in data/processed/.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved successfully to: {output_path}")


def run_pipeline():
    """
    Runs the complete data preparation and enrichment workflow.
    """
    raw_df = load_raw_data()
    enriched_df = enrich_dataset(raw_df)
    save_processed_data(enriched_df)
    
    # Print quick summary for verification
    print("\n--- Pipeline Summary ---")
    print(f"Total Orders Processed: {len(enriched_df):,}")
    print(f"Average Lead Time: {enriched_df['Lead Time (Days)'].mean():.1f} days")
    print(f"Average Current Distance: {enriched_df['Transit Distance (Miles)'].mean():.1f} miles")
    print(f"Average Optimal Distance: {enriched_df['Minimum Distance (Miles)'].mean():.1f} miles")
    print(f"Orders where closer factory exists: {enriched_df['Is Closer Factory Available'].sum():,} ({enriched_df['Is Closer Factory Available'].mean()*100:.1f}%)")
    print(f"Total Potential Mileage Saved: {enriched_df['Potential Distance Saved (Miles)'].sum():,.0f} miles")
    return enriched_df


if __name__ == "__main__":
    run_pipeline()
