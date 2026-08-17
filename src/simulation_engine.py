"""
Scenario Simulation Engine for Nassau Candy
--------------------------------------------
This module allows operations and leadership to run 'What-If' simulations:
For any chosen product, destination state/city, and shipping mode,
it simulates manufacturing and shipping the order from each of the 5 factories.

It computes:
1. Exact Transit Distance (Miles) from each factory.
2. Predicted Delivery Lead Time (Days) using the trained ML model.
3. Lead Time Reduction (%) compared to the baseline factory.
4. Distance Saved (%) and estimated freight cost impact.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geo_utils import (
    FACTORIES,
    PRODUCT_FACTORY_MAP,
    get_customer_coordinates,
    calculate_haversine_distance,
    get_distance_to_factory
)
from src.model_engine import load_trained_model


def get_product_division(product_name):
    """
    Returns division (Chocolate, Sugar, or Other) for a given product.
    """
    if "Wonka Bar" in product_name:
        return "Chocolate"
    elif product_name in ["Laffy Taffy", "SweeTARTS", "Nerds", "Fun Dip", "Everlasting Gobstopper", "Hair Toffee"]:
        return "Sugar"
    else:
        return "Other"


def get_state_region(state_name):
    """
    Maps US State to one of the 4 sales regions in the dataset:
    Pacific, Atlantic, Interior, Gulf.
    """
    pacific_states = ['California', 'Washington', 'Oregon', 'Nevada', 'Arizona', 'Idaho', 'Utah', 'Alaska', 'Hawaii']
    atlantic_states = ['New York', 'Pennsylvania', 'New Jersey', 'Massachusetts', 'Virginia', 'North Carolina', 'South Carolina', 'Georgia', 'Maryland', 'Connecticut', 'Delaware', 'District of Columbia', 'Maine', 'New Hampshire', 'Rhode Island', 'Vermont', 'West Virginia']
    gulf_states = ['Texas', 'Florida', 'Louisiana', 'Alabama', 'Mississippi']
    
    if state_name in pacific_states:
        return 'Pacific'
    elif state_name in atlantic_states:
        return 'Atlantic'
    elif state_name in gulf_states:
        return 'Gulf'
    else:
        return 'Interior'


def simulate_all_factories_for_order(
    product_name,
    state,
    city="New York City",
    ship_mode="Standard Class",
    units=3,
    sales=12.0,
    cost=4.0,
    month=6,
    model_pipeline=None
):
    """
    Simulates fulfilling an order from all 5 candidate factories.
    
    Returns:
    - Comparison DataFrame ranking all 5 factories with distances,
      predicted lead times, and savings.
    """
    if model_pipeline is None:
        model_pipeline = load_trained_model()

    # Determine default current factory and customer info
    current_factory = PRODUCT_FACTORY_MAP.get(product_name, "Lot's O' Nuts")
    division = get_product_division(product_name)
    region = get_state_region(state)
    cust_lat, cust_lon = get_customer_coordinates(city, state)

    # Calculate current baseline distance
    current_f_lat = FACTORIES[current_factory]["latitude"]
    current_f_lon = FACTORIES[current_factory]["longitude"]
    current_distance = calculate_haversine_distance(current_f_lat, current_f_lon, cust_lat, cust_lon)

    # Baseline prediction vector
    baseline_row = pd.DataFrame([{
        'Product Name': product_name,
        'Division': division,
        'Current Factory': current_factory,
        'Region': region,
        'State/Province': state,
        'Ship Mode': ship_mode,
        'Transit Distance (Miles)': current_distance,
        'Units': units,
        'Sales': sales,
        'Cost': cost,
        'Order Month': month
    }])
    baseline_pred_lead_time = float(model_pipeline.predict(baseline_row)[0])

    simulation_results = []

    for candidate_factory, factory_info in FACTORIES.items():
        # Calculate transit distance from this candidate factory
        cand_lat = factory_info["latitude"]
        cand_lon = factory_info["longitude"]
        distance_miles = calculate_haversine_distance(cand_lat, cand_lon, cust_lat, cust_lon)

        # Prepare feature row for candidate factory
        feature_row = pd.DataFrame([{
            'Product Name': product_name,
            'Division': division,
            'Current Factory': candidate_factory,
            'Region': region,
            'State/Province': state,
            'Ship Mode': ship_mode,
            'Transit Distance (Miles)': distance_miles,
            'Units': units,
            'Sales': sales,
            'Cost': cost,
            'Order Month': month
        }])

        predicted_lead_time = float(model_pipeline.predict(feature_row)[0])
        
        # Calculate improvements vs baseline
        dist_saved = round(current_distance - distance_miles, 1)
        dist_saved_pct = round((dist_saved / current_distance) * 100, 1) if current_distance > 0 else 0.0
        
        lead_time_saved = round(baseline_pred_lead_time - predicted_lead_time, 1)
        lead_time_saved_pct = round((lead_time_saved / baseline_pred_lead_time) * 100, 1) if baseline_pred_lead_time > 0 else 0.0

        # Estimated freight cost scaling factor based on distance
        # Standard freight rate: shorter distance reduces variable transit cost
        freight_cost_factor = distance_miles / max(current_distance, 1.0)
        est_unit_cost = round(cost * (0.85 + 0.15 * freight_cost_factor), 2)
        est_gross_profit = round(sales - est_unit_cost, 2)
        est_margin_pct = round((est_gross_profit / sales) * 100, 1) if sales > 0 else 0.0

        simulation_results.append({
            "Factory": candidate_factory,
            "Hub Location": f"{factory_info['city']}, {factory_info['state']}",
            "Is Current Assignment": (candidate_factory == current_factory),
            "Distance (Miles)": round(distance_miles, 1),
            "Distance Saved (Miles)": dist_saved,
            "Distance Saved (%)": dist_saved_pct,
            "Predicted Lead Time (Days)": round(predicted_lead_time, 1),
            "Lead Time Saved (Days)": lead_time_saved,
            "Lead Time Saved (%)": lead_time_saved_pct,
            "Est. Cost ($)": est_unit_cost,
            "Est. Gross Profit ($)": est_gross_profit,
            "Est. Gross Margin (%)": est_margin_pct
        })

    results_df = pd.DataFrame(simulation_results)
    
    # Sort by closest distance and highest lead time savings
    results_df = results_df.sort_values(by="Distance (Miles)", ascending=True).reset_index(drop=True)
    return results_df, baseline_pred_lead_time, current_distance


if __name__ == "__main__":
    # Test simulation query
    print("Testing Factory Reallocation Simulation...")
    sample_product = "Wonka Bar - Milk Chocolate"
    sample_state = "California"
    
    sim_df, base_time, base_dist = simulate_all_factories_for_order(
        product_name=sample_product,
        state=sample_state,
        city="Los Angeles",
        ship_mode="Standard Class"
    )
    
    print(f"\nProduct: {sample_product} | Destination: Los Angeles, {sample_state}")
    print(f"Baseline Lead Time: {base_time:.1f} days | Baseline Distance: {base_dist:.1f} miles\n")
    print(sim_df[['Factory', 'Distance (Miles)', 'Distance Saved (Miles)', 'Predicted Lead Time (Days)', 'Lead Time Saved (%)']].to_string(index=False))
