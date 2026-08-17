# Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor

> **Technical Documentation & Project Guidelines**  
> *Unified Mentor Machine Learning Internship Project*

---

## IMPORTANT DISCLAIMER & NOTICE OF NON-AFFILIATION

> [!IMPORTANT]
> **Academic, Educational & Evaluation Purpose Only:**
> - **Internship Project Only:** This documentation and all associated materials are created strictly for completing the **Unified Mentor Machine Learning Internship Project**.
> - **No Proprietary Claim:** The author does not claim ownership or rights over the *Nassau Candy Distributor* business case, trademarks, or datasets. All intellectual property remains with the respective owners / Unified Mentor.
> - **Public Repository Notice:** This repository is hosted publicly solely to allow submission and review by Unified Mentor evaluators.
> - **No Commercial Rights:** No commercial sale, redistribution, or monetization of this material is permitted.
> - **No Legal Liability:** All materials are provided **"AS IS"** for educational demonstration. The author accepts **no legal liability or responsibility** for third-party use or interpretation.

---

## 1. Background and Context

**Nassau Candy Distributor** is a leading wholesale manufacturer and distributor of confectionery and specialty foods across the United States.

However, leadership faces critical strategic questions:
* *What operational and distribution changes should be implemented to improve delivery performance?*
* *How can shipping lead times and freight costs be reduced while preserving profitability?*

This project introduces a **Decision Intelligence System** that:
1. **Predicts** shipping outcomes under different fulfillment configurations.
2. **Recommends** which products should be reassigned to alternative factories based on customer geography.
3. **Balances** shipping efficiency (speed & distance) and financial profitability.

---

## 2. Problem Statement

Nassau Candy currently assigns products to factories using **static rules and legacy processes**. This leads to:
* **Suboptimal Shipping Distances**: High cross-country freight mileage when products are manufactured far from customer demand clusters.
* **High Lead Times for Certain Regions**: Distribution bottlenecks in high-volume regions.
* **Margin Erosion**: Unnecessary transportation and expediting costs that reduce gross margins.

### Current Gaps:
* No automated system to **simulate** factory-product reassignment scenarios.
* Inability to **quantify the operational and financial impact** before execution.
* Lack of an algorithmic framework to **recommend optimal configurations at scale**.

---

## 3. Dataset Fields Description

| Field Name | Description | Example / Notes |
| :--- | :--- | :--- |
| **`Row ID`** | Unique row identifier | `1, 2, 3, ...` |
| **`Order ID`** | Unique order identifier | `US-2021-103800-CHO-MIL-31000` |
| **`Order Date`** | Date of order | `03-01-2024` (DD-MM-YYYY) |
| **`Ship Date`** | Date of shipment | `30-06-2026` (DD-MM-YYYY) |
| **`Ship Mode`** | Shipping method of order | `Standard Class`, `Second Class`, `First Class`, `Same Day` |
| **`Customer ID`** | Unique customer identifier | `103800` |
| **`Country/Region`** | Country or region of customer | `United States` |
| **`City`** | City of customer | `Houston`, `Naperville`, `Los Angeles`, etc. |
| **`State/Province`** | State/province of customer | `Texas`, `Illinois`, `California`, etc. |
| **`Postal Code`** | Postal code / ZIP code of customer | `77095`, `60540`, etc. |
| **`Division`** | Product division | `Chocolate`, `Sugar`, `Other` |
| **`Region`** | Geographic region of customer | `Interior`, `Atlantic`, `Gulf`, `Pacific` |
| **`Product ID`** | Unique product SKU identifier | `CHO-MIL-31000`, `CHO-NUT-13000`, etc. |
| **`Product Name`** | Product long name | `Wonka Bar - Milk Chocolate`, etc. |
| **`Sales`** | Total sales revenue of order | Float ($) |
| **`Units`** | Total units ordered | Integer |
| **`Gross Profit`** | Gross profit of order (Sales - Cost) | Float ($) |
| **`Cost`** | Cost to manufacture / fulfill | Float ($) |

---

## 4. Factory Coordinates Master Reference

| Factory Name | Latitude | Longitude | Primary Regional Hub |
| :--- | :---: | :---: | :--- |
| **`Lot's O' Nuts`** | `32.881893` | `-111.768036` | Casa Grande / Southwest (AZ) |
| **`Wicked Choccy's`** | `32.076176` | `-81.088371` | Savannah / Southeast (GA) |
| **`Sugar Shack`** | `48.119140` | `-96.181150` | Thief River Falls / Upper Midwest (MN) |
| **`Secret Factory`** | `41.446333` | `-90.565487` | Rock Island / Central Midwest (IL) |
| **`The Other Factory`** | `35.117500` | `-89.971107` | Memphis / Mid-South (TN) |

---

## 5. Products and Baseline Factory Correlation

| Division | Product Name | Current Assigned Factory |
| :--- | :--- | :--- |
| **Chocolate** | Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts |
| **Chocolate** | Wonka Bar - Fudge Mallows | Lot's O' Nuts |
| **Chocolate** | Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts |
| **Chocolate** | Wonka Bar - Milk Chocolate | Wicked Choccy's |
| **Chocolate** | Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's |
| **Sugar** | Laffy Taffy | Sugar Shack |
| **Sugar** | SweeTARTS | Sugar Shack |
| **Sugar** | Nerds | Sugar Shack |
| **Sugar** | Fun Dip | Sugar Shack |
| **Other** | Fizzy Lifting Drinks | Sugar Shack |
| **Sugar** | Everlasting Gobstopper | Secret Factory |
| **Other** | Lickable Wallpaper | Secret Factory |
| **Other** | Wonka Gum | Secret Factory |
| **Sugar** | Hair Toffee | The Other Factory |
| **Other** | Kazookles | The Other Factory |

---

## 6. Analytical Methodology (Step-by-Step)

```
[ Data Preparation & Encoding ] ---> [ Route & Bottleneck Clustering ] ---> [ Predictive Lead Time ML ]
                                                                                   |
[ Streamlit Web Application ] <--- [ Optimization & Top-N Recommendations ] <-------+---> [ Scenario Simulator Engine ]
```

### 6.1 Data Preparation & Encoding
* Calculate **Lead Time** in days (Ship Date - Order Date).
* Compute **Geodesic / Haversine Distance** between factory coordinates and destination city/state coordinates.
* Normalize numerical features (`Distance`, `Units`, `Sales`, `Cost`).
* Encode categorical variables (`Region`, `Ship Mode`, `Division`, `Product Name`).
* Remove extreme outliers and validate consistency.
* Create training-ready feature matrix.

### 6.2 Route & Product Clustering
* Cluster shipping routes by performance similarity (e.g., using **K-Means / DBSCAN**).
* Identify:
  * Consistently slow and high-latency routes.
  * Congested region-product combinations causing operational drag.

### 6.3 Predictive Modeling Objective
Predict expected shipping lead time given:
* `Product Name`
* `Origin Factory`
* `Destination Region / Coordinates`
* `Ship Mode`

#### Supervised ML Models:
1. **Linear Regression** (Baseline benchmark)
2. **Random Forest Regressor** (Non-linear ensemble)
3. **Gradient Boosting Regressor** (XGBoost / LightGBM / CatBoost)

#### Model Evaluation Metrics:
* **RMSE (Root Mean Squared Error)**: Penalizes large delivery delay errors.
* **MAE (Mean Absolute Error)**: Measures average prediction error in days.
* **R-squared Score**: Quantifies proportion of variance explained by the model.
* *Selection criteria: Best-performing model based on accuracy + interpretability.*

### 6.4 Scenario Simulation Engine
For each product order profile:
* Simulate assignment to all alternate candidate factories.
* Predict new delivery lead times using the trained ML model.
* Estimate operational distance reduction and transit time improvement.
* Measure profit sensitivity and cost implications.

### 6.5 Optimization & Recommendation Logics
Rank factory options using multi-objective scoring:
* **Lead time reduction (%)** (Speed gain)
* **Risk reduction** (Prediction variance / reliability)
* **Profit impact** (Financial safety)
* Generate **Top-N Factory Reassignment Recommendations** for leadership.

---

## 7. Key Performance Indicators (KPIs)

| KPI Metric | Focus Area | Operational Target |
| :--- | :--- | :--- |
| **Lead Time Reduction (%)** | Operational gain | >= 20% reduction on bottleneck routes |
| **Profit Impact Stability** | Financial safety | Protect and maintain positive gross profit margins |
| **Scenario Confidence Score** | Prediction reliability | High statistical confidence (R-squared / low RMSE) |
| **Recommendation Coverage** | Scalability | 100% SKU and regional route coverage |

---

## 8. Streamlit Web Application Requirements

### Dashboard Modules:
1. **Factory Optimization Simulator**:
   * Select product, destination region, and shipping mode.
   * View predicted performance and lead times across all 5 factories.
2. **What-If Scenario Analysis**:
   * Compare current vs. recommended factory assignments.
   * Visualize lead-time improvements, route maps, and distributions.
3. **Recommendation Dashboard**:
   * Ranked reassignment suggestions.
   * Quantified expected efficiency gains and mileage savings.
4. **Risk & Impact Panel**:
   * Profit impact alerts.
   * High-risk reassignment warnings and capacity flags.

### User Capabilities & Controls:
* **Product Selector** (Dropdown of all 15 SKUs)
* **Region / State Selector** (Geographic destination filters)
* **Ship Mode Filter** (`Standard Class`, `Second Class`, `First Class`, `Same Day`)
* **Optimization Priority Slider** (Interactive weighting: *Speed Priority vs. Profit Priority*)

---

## 9. Deliverables and Submission Checklist

* [x] **Research Paper / Technical Report**: In-depth analysis containing Exploratory Data Analysis (EDA), mathematical modeling, ML benchmark tables, simulation findings, and strategic business recommendations.
* [x] **Streamlit Web Application**: Live interactive dashboard meeting all dashboard module requirements.
* [x] **Executive Summary**: High-impact brief and presentation deck tailored for executive and operational stakeholders.
* [x] **Clean Codebase & Notebooks**: Well-structured Python project with reproducible pipelines.

---

## 10. Conclusion

This project elevates **Nassau Candy Distributor** from descriptive historical reporting into **prescriptive decision intelligence**. By combining predictive machine learning models with multi-objective optimization algorithms, it delivers actionable factory reallocation recommendations that significantly improve shipping efficiency without compromising profitability.
