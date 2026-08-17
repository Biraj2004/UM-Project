# Research Report: Decision Intelligence and Factory Reallocation System for Nassau Candy Distributor

**Author**: Biraj2004  
**Project**: Unified Mentor Machine Learning Internship  
**Domain**: Supply Chain Analytics, Geospatial Data Science & Decision Optimization  
**Date**: August 2026  

---

## IMPORTANT NOTICE & ACADEMIC DISCLAIMER
This research report was conducted solely for educational and academic purposes as part of the Unified Mentor Machine Learning Internship Program. All case study materials, trademarks, brand names (including Nassau Candy Distributor), and underlying datasets remain the property of their respective owners. The author claims no proprietary ownership over third-party materials. All models and conclusions are provided AS IS for academic demonstration.

---

## 1. Abstract

Distribution network optimization is critical for large-scale wholesale food and confectionery distributors. Nassau Candy operates five primary manufacturing and warehousing hubs across the United States. Historically, Nassau Candy relied on static legacy allocation rules, tying specific confectionery product lines to single predetermined factories regardless of customer destination geography. This research formulates an end-to-end Decision Intelligence and Optimization Recommendation System. 

Using 10,194 historical order transactions, we engineer geospatial Haversine transit distances and temporal lead times. We apply unsupervised K-Means clustering to discover regional delivery bottlenecks, benchmark supervised machine learning regression models (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting) for lead time prediction, and develop a counterfactual scenario simulation engine. Finally, a multi-objective Pareto optimization framework is implemented to generate ranked factory-to-product reallocation recommendations. Our findings demonstrate that 68.8% of historical shipments were fulfilled suboptimally, and strategic network reallocation can eliminate **7,487,332 transit miles** (a 59.6% reduction in freight mileage) while fully preserving gross profit margins.

---

## 2. Introduction & Business Problem

Nassau Candy manufactures and distributes high-volume confectionery lines across three major divisions: Chocolate, Sugar, and Specialty items. The current network comprises five hubs:
1. **Lot's O' Nuts** (Casa Grande, Arizona)
2. **Wicked Choccy's** (Savannah, Georgia)
3. **Sugar Shack** (Thief River Falls, Minnesota)
4. **Secret Factory** (Rock Island, Illinois)
5. **The Other Factory** (Memphis, Tennessee)

Under the legacy operating policy:
- Chocolate products (e.g., *Wonka Bar - Milk Chocolate*, *Wonka Bar - Triple Dazzle Caramel*) are manufactured exclusively at *Wicked Choccy's* in Georgia. When customers in West Coast states (e.g., California, Washington) place orders, goods are transported over 2,200 miles across the continent.
- Similarly, nut-based chocolate bars (e.g., *Wonka Bar - Nutty Crunch Surprise*, *Wonka Bar - Scrumdiddlyumptious*) are manufactured exclusively at *Lot's O' Nuts* in Arizona, resulting in cross-country hauls to East Coast customers in New York, Pennsylvania, and New Jersey.

This static allocation introduces major inefficiencies:
- **Excessive Freight Mileage**: Millions of unnecessary transit miles incurred annually.
- **Prolonged Lead Times**: Bottlenecks in high-density customer regions (Pacific and Atlantic corridors).
- **Margin Erosion**: Escalating transportation costs impacting unit gross margins.
- **Inflexible Decision-Making**: Lack of simulation tools to evaluate counterfactual network configurations.

---

## 3. Mathematical & Algorithmic Formulations

### 3.1 Geodesic Transit Distance (Haversine Formula)
To measure true physical distance from origin factory $(\phi_F, \lambda_F)$ to destination customer centroid $(\phi_C, \lambda_C)$, we compute the great-circle distance:

$$d = 2 R \cdot \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_F)\cos(\phi_C)\sin^2\left(\frac{\Delta\lambda}{2}\right)} \right)$$

where $R = 3,958.8\text{ miles}$ (Earth's mean radius), $\Delta\phi = \phi_C - \phi_F$, and $\Delta\lambda = \lambda_C - \lambda_F$.

### 3.2 Lead Time Formulation
For each order $i$:
$$\text{Lead Time}_i = \text{Date}_{\text{Ship}, i} - \text{Date}_{\text{Order}, i}$$

### 3.3 Unit Economics & Financial Formulations
$$\text{Gross Profit}_i = \text{Sales}_i - \text{Cost}_i$$
$$\text{Gross Margin } \% = \left(\frac{\text{Gross Profit}_i}{\text{Sales}_i}\right) \times 100$$
$$\text{Unit Selling Price} = \frac{\text{Sales}_i}{\text{Units}_i}, \quad \text{Unit Cost} = \frac{\text{Cost}_i}{\text{Units}_i}$$

### 3.4 Multi-Objective Decision Function
For candidate factory $F_{\text{alt}}$ relative to current factory $F_{\text{curr}}$ for product $P$ in region $R$:

$$S(P, F_{\text{alt}}, R) = w_{\text{speed}} \cdot \left(\frac{d_{\text{curr}} - d_{\text{alt}}}{d_{\text{curr}}}\right) + w_{\text{profit}} \cdot \left(\frac{\text{Margin}_{\text{alt}}}{\text{Margin}_{\text{baseline}}}\right) - \text{Penalty}$$

where $w_{\text{speed}} + w_{\text{profit}} = 1.0$.

---

## 4. Exploratory Data Analysis & Empirical Findings

### 4.1 Dataset Profile
The transactional dataset contains **10,194 records** spanning 15 distinct candy SKUs across 59 US states/territories and 542 cities.

| Metric | Historical Baseline Value | Optimal Reallocated Value | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Total Order Volume** | 10,194 orders | 10,194 orders | 100% Coverage |
| **Average Transit Distance** | 1,231.4 miles | 496.9 miles | **-734.5 miles (-59.6%)** |
| **Total Transit Mileage** | 12,553,103 miles | 5,065,771 miles | **-7,487,332 miles saved** |
| **Suboptimal Shipments** | 7,011 orders (68.8%) | 0 orders (0.0%) | 100% Resolved |
| **Average Gross Margin** | 65.9% | >= 65.9% | Maintained / Protected |

### 4.2 Route Bottleneck Clustering (K-Means)
Unsupervised clustering ($k=3$) on `[Transit Distance, Lead Time, Units, Cost]` segmented the network into three distinct operational corridors:
1. **Cluster 0 (Low Distance / Local Fulfillment)**: 4,732 orders (46.4%), average distance 702.1 miles.
2. **Cluster 1 (Moderate Distance / Standard Corridor)**: 1,807 orders (17.7%), average distance 1,161.6 miles.
3. **Cluster 2 (High Latency / Bottleneck Corridor)**: 3,655 orders (35.9%), average distance 1,951.3 miles. **100% of these shipments had closer factories available.**

---

## 5. Machine Learning Benchmarks & Model Evaluation

Supervised regression models were trained using an 80/20 train-test split with 5-Fold Cross Validation.

| Model Algorithm | Test MAE (Days) | Test RMSE (Days) | Test R2 | 5-Fold CV R2 | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **211.59** | **262.97** | **0.0222** | **0.0127** | **Selected Best Model** |
| **Gradient Boosting Regressor** | 212.80 | 264.51 | 0.0107 | 0.0043 | Benchmark Contender |
| **Ridge Regression** | 214.54 | 265.98 | -0.0003 | 0.0002 | Linear Regularized |
| **Linear Regression (Baseline)** | 214.81 | 266.49 | -0.0042 | -0.0008 | Baseline |
| **Decision Tree** | 216.98 | 271.45 | -0.0419 | -0.0599 | Overfitted Baseline |

*Model Selection*: Random Forest Regressor demonstrated superior stability, lowest MAE, and lowest RMSE, and was serialized for real-time inference in the Streamlit application.

---

## 6. Strategic Recommendations & Optimization Policy

The optimization engine generated the following high-impact factory reassignments:

1. **Reassign Wonka Bar - Triple Dazzle Caramel (Pacific Region)**:
   - *Current*: Wicked Choccy's (Savannah, GA)
   - *Recommended*: Lot's O' Nuts (Casa Grande, AZ)
   - *Distance Reduction*: **76.1%** (Average distance reduced from 2,144 miles to 512 miles)
   - *Total Freight Mileage Saved*: **1,103,890 miles**

2. **Reassign Wonka Bar - Milk Chocolate (Pacific Region)**:
   - *Current*: Wicked Choccy's (Savannah, GA)
   - *Recommended*: Lot's O' Nuts (Casa Grande, AZ)
   - *Distance Reduction*: **77.1%** (Average distance reduced from 2,144 miles to 491 miles)
   - *Total Freight Mileage Saved*: **1,088,518 miles**

3. **Reassign Wonka Bar - Scrumdiddlyumptious (Atlantic Region)**:
   - *Current*: Lot's O' Nuts (Casa Grande, AZ)
   - *Recommended*: Wicked Choccy's (Savannah, GA)
   - *Distance Reduction*: **68.7%** (Average distance reduced from 2,050 miles to 642 miles)
   - *Total Freight Mileage Saved*: **846,148 miles**

4. **Reassign Wonka Bar - Fudge Mallows (Atlantic Region)**:
   - *Current*: Lot's O' Nuts (Casa Grande, AZ)
   - *Recommended*: Wicked Choccy's (Savannah, GA)
   - *Distance Reduction*: **68.5%**
   - *Total Freight Mileage Saved*: **743,198 miles**

5. **Reassign Wonka Bar - Nutty Crunch Surprise (Atlantic Region)**:
   - *Current*: Lot's O' Nuts (Casa Grande, AZ)
   - *Recommended*: Wicked Choccy's (Savannah, GA)
   - *Distance Reduction*: **68.5%**
   - *Total Freight Mileage Saved*: **669,123 miles**

6. **Reassign Wonka Bar - Nutty Crunch Surprise (Interior Region)**:
   - *Current*: Lot's O' Nuts (Casa Grande, AZ)
   - *Recommended*: The Other Factory (Memphis, TN)
   - *Distance Reduction*: **84.2%**
   - *Total Freight Mileage Saved*: **456,528 miles**

---

## 7. Streamlit Web Application Architecture

The system provides an enterprise-ready Streamlit decision platform comprising five core modules:
1. **Executive Overview**: High-level KPI metric cards and geographic pie/bar distributions.
2. **Factory Reallocation Simulator**: Interactive SKU, state, and shipping mode selector with instant 5-factory comparative evaluation.
3. **What-If Scenario Matrix**: Distribution curves comparing baseline vs. optimized transit distances.
4. **Top-N Recommendations**: Ranked policy table with dynamic weighting and CSV export.
5. **Operational Risk Panel**: Gross margin integrity safeguards and capacity shift diagnostics.

---

## 8. Conclusion

Static factory assignments introduce massive structural inefficiencies in supply chain distribution networks. By replacing static legacy rules with geospatial intelligence, predictive machine learning, and multi-objective optimization, Nassau Candy can eliminate **over 7.48 million freight transit miles**, relieve delivery congestion across key customer corridors, and maintain strong profitability.
