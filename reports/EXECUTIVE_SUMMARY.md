# Executive Summary: Nassau Candy Shipping Optimization & Factory Reallocation System

**Prepared for**: Executive Leadership, Operations Stakeholders & Project Evaluators  
**Project**: Unified Mentor Machine Learning Internship  
**Author**: Biraj2004  
**Date**: August 2026  

---

## IMPORTANT NOTICE
This summary is prepared strictly for educational and internship evaluation purposes under the Unified Mentor Machine Learning Internship Program. All case study concepts, brand names, and datasets remain the property of their respective owners. Code and insights are provided AS IS.

---

## 1. Strategic Context & Challenge

Nassau Candy Distributor fulfills orders across the continental United States from five manufacturing factories. However, product assignments to factories were historically determined by static rules rather than customer demand geography:
* **The Inefficiency**: Over **68.8% of historical orders** were shipped from a distant factory when a closer production facility was operational.
* **The Cost**: Cross-country shipping inflated freight mileage to **12.55 million miles**, generating delivery lead time bottlenecks and eroding profit margins.

---

## 2. Decision Intelligence Solution

We developed an end-to-end Machine Learning & Multi-Objective Optimization System that:
1. **Computes Exact Geospatial Distance**: Calculates physical transit distance from each factory to customer destination coordinates.
2. **Clusters Route Bottlenecks**: Identifies high-latency routes using unsupervised K-Means clustering.
3. **Predicts Fulfillment Performance**: Uses a trained Random Forest Regressor to forecast delivery lead times under any configuration.
4. **Simulates Reallocations**: Allows real-time counterfactual testing across all 5 candidate factories.
5. **Recommends Optimal Policies**: Balances delivery speed and margin preservation via multi-objective Pareto optimization.

---

## 3. Key Quantitative Results

| Key Metric | Current Baseline | Optimized Policy | Net Impact |
| :--- | :--- | :--- | :--- |
| **Average Freight Distance** | 1,231 miles / order | 497 miles / order | **-59.6% reduction** |
| **Total Freight Mileage** | 12,553,103 miles | 5,065,771 miles | **7,487,332 miles saved** |
| **Suboptimal Shipments** | 68.8% of total volume | 0.0% of total volume | **100% optimized** |
| **Average Gross Profit Margin** | 65.9% | >= 65.9% | **Protected / Enhanced** |

---

## 4. Top Recommended Actions

1. **Reassign Wonka Bar - Milk Chocolate & Triple Dazzle Caramel in the Pacific Region to Lot's O' Nuts (AZ)**:
   - Eliminates over **2.19 million miles** of cross-country hauling from Georgia.
   - Reduces transit distance by **>76%** for West Coast customers.
2. **Reassign Wonka Bar - Scrumdiddlyumptious, Fudge Mallows & Nutty Crunch in the Atlantic Region to Wicked Choccy's (GA)**:
   - Eliminates over **2.25 million miles** of cross-country hauling from Arizona.
   - Reduces transit distance by **>68%** for East Coast customers.
3. **Reassign Interior Region Nutty Crunch Orders to The Other Factory (TN)**:
   - Cuts transit distance by **84.2%** for midwestern/southern customers.

---

## 5. Software Deliverables

1. **Interactive Web Application**: `app/app.py` (Streamlit Dashboard with Simulator, What-If Matrix, and Downloadable Policy Recommendations).
2. **Reproducible Python Pipelines**: Clean, modular code in `src/` (`data_pipeline.py`, `geo_utils.py`, `clustering.py`, `model_engine.py`, `simulation_engine.py`, `optimization_engine.py`).
3. **Comprehensive Research Report**: `reports/RESEARCH_PAPER.md`.
