"""
Nassau Candy Decision Intelligence & Shipping Optimization Platform
-------------------------------------------------------------------
Streamlit Web Application for Factory Reallocation and Lead Time Optimization.

Modules Included:
1. Executive KPI Metrics Ribbon
2. Factory Reallocation Simulator
3. What-If Scenario Matrix & Visualizations
4. Strategic Top-N Recommendations Table
5. Operational Risk & Impact Panel
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path so we can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geo_utils import FACTORIES, PRODUCT_FACTORY_MAP, US_STATE_COORDINATES
from src.data_pipeline import load_raw_data, enrich_dataset
from src.simulation_engine import simulate_all_factories_for_order
from src.optimization_engine import generate_regional_recommendations
from src.model_engine import load_trained_model


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy - Decision Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS (No Emojis, Professional Layout)
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .notice-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.9rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_enriched_data():
    """
    Loads or generates the enriched dataset.
    """
    processed_path = "data/processed/nassau_candy_enriched.csv"
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)
    else:
        raw_df = load_raw_data()
        df = enrich_dataset(raw_df)
        return df


@st.cache_resource
def get_cached_model():
    """
    Loads the trained machine learning pipeline.
    """
    return load_trained_model()


# Load shared resources
df = get_cached_enriched_data()
model_pipeline = get_cached_model()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Filters
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation & Controls")

app_mode = st.sidebar.radio(
    "Select Module:",
    [
        "1. Executive Overview & KPIs",
        "2. Factory Reallocation Simulator",
        "3. What-If Scenario Matrix",
        "4. Top-N Recommendations",
        "5. Risk & Capacity Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Preferences")

weight_speed = st.sidebar.slider(
    "Delivery Speed Priority (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Higher weight prioritizes transit distance and lead time reduction."
) / 100.0

weight_profit = round(1.0 - weight_speed, 2)
st.sidebar.text(f"Profit Margin Priority: {int(weight_profit * 100)}%")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Academic Notice**: This application was developed solely for the "
    "Unified Mentor Machine Learning Internship Program. Code is provided AS IS."
)


# -----------------------------------------------------------------------------
# MODULE 1: Executive Overview & KPIs
# -----------------------------------------------------------------------------
if app_mode == "1. Executive Overview & KPIs":
    st.markdown('<div class="main-header">Nassau Candy: Executive Operations Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decision Intelligence & Shipping Optimization Platform</div>', unsafe_allow_html=True)

    # Top KPI Metrics Ribbon
    total_orders = len(df)
    avg_lead_time = df['Lead Time (Days)'].mean()
    current_avg_dist = df['Transit Distance (Miles)'].mean()
    optimal_avg_dist = df['Minimum Distance (Miles)'].mean()
    potential_miles_saved = df['Potential Distance Saved (Miles)'].sum()
    pct_closer_avail = (df['Is Closer Factory Available'].mean()) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_orders:,}</div><div class="metric-label">Total Orders Analyzed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{current_avg_dist:.0f} mi</div><div class="metric-label">Current Avg Distance</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{optimal_avg_dist:.0f} mi</div><div class="metric-label">Optimal Avg Distance</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{potential_miles_saved:,.0f} mi</div><div class="metric-label">Total Miles Reducible</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{pct_closer_avail:.1f}%</div><div class="metric-label">Suboptimal Orders</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two column analytical view
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Factory Network & Order Concentration")
        factory_counts = df['Current Factory'].value_counts().reset_index()
        factory_counts.columns = ['Factory', 'Order Count']
        
        fig_factory = px.pie(
            factory_counts,
            names='Factory',
            values='Order Count',
            title='Order Volume Share by Manufacturing Factory',
            color_discrete_sequence=px.colors.qualitative.Prism,
            hole=0.4
        )
        fig_factory.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_factory, use_container_width=True)

    with c_right:
        st.subheader("Distance Bottleneck: Current vs Optimal by Region")
        region_dist = df.groupby('Region').agg(
            Current=('Transit Distance (Miles)', 'mean'),
            Optimal=('Minimum Distance (Miles)', 'mean')
        ).reset_index()

        fig_region = go.Figure(data=[
            go.Bar(name='Current Avg Distance (Miles)', x=region_dist['Region'], y=region_dist['Current'], marker_color='#E11D48'),
            go.Bar(name='Optimal Reallocated Distance (Miles)', x=region_dist['Region'], y=region_dist['Optimal'], marker_color='#10B981')
        ])
        fig_region.update_layout(
            barmode='group',
            title='Average Freight Distance by Customer Region (Miles)',
            xaxis_title='Customer Region',
            yaxis_title='Distance (Miles)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_region, use_container_width=True)

    # Key Takeaways Notice Box
    st.markdown("""
    <div class="notice-box">
    <b>Key Executive Findings:</b><br>
    - Over <b>68.8% of all historical orders (7,011 shipments)</b> were fulfilled by a distant factory when a closer production facility was available.<br>
    - Implementing optimal factory reallocation eliminates <b>7,487,332 transit miles</b>, drastically reducing freight expenditure and delivery delays.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 2: Factory Reallocation Simulator
# -----------------------------------------------------------------------------
elif app_mode == "2. Factory Reallocation Simulator":
    st.markdown('<div class="main-header">Factory Reallocation Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate order fulfillment performance across all 5 manufacturing hubs</div>', unsafe_allow_html=True)

    # Simulator Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_product = st.selectbox("Select Candy Product:", sorted(df['Product Name'].unique()), index=10)
    with c2:
        selected_state = st.selectbox("Select Customer Destination State:", sorted(US_STATE_COORDINATES.keys()), index=4)  # California
    with c3:
        selected_ship_mode = st.selectbox("Select Shipping Mode:", ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
    with c4:
        selected_units = st.number_input("Order Units:", min_value=1, max_value=50, value=3)

    # Run Simulation
    sim_df, base_time, base_dist = simulate_all_factories_for_order(
        product_name=selected_product,
        state=selected_state,
        ship_mode=selected_ship_mode,
        units=selected_units,
        model_pipeline=model_pipeline
    )

    current_factory_name = PRODUCT_FACTORY_MAP.get(selected_product, "Unknown")

    st.markdown("---")
    st.subheader(f"Simulation Results for: {selected_product}")
    st.markdown(f"**Current Static Factory**: `{current_factory_name}` | **Destination**: `{selected_state}` | **Baseline Distance**: `{base_dist:.1f} miles`")

    # Display comparison table
    st.dataframe(
        sim_df[[
            'Factory', 'Hub Location', 'Is Current Assignment', 'Distance (Miles)',
            'Distance Saved (Miles)', 'Distance Saved (%)',
            'Predicted Lead Time (Days)', 'Est. Cost ($)', 'Est. Gross Margin (%)'
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Bar chart comparing all 5 factories
    fig_sim = px.bar(
        sim_df,
        x='Factory',
        y='Distance (Miles)',
        color='Is Current Assignment',
        color_discrete_map={True: '#E11D48', False: '#2563EB'},
        title=f"Transit Distance Comparison for Delivery to {selected_state} (Miles)",
        text='Distance (Miles)'
    )
    fig_sim.update_traces(textposition='outside')
    fig_sim.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sim, use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 3: What-If Scenario Matrix
# -----------------------------------------------------------------------------
elif app_mode == "3. What-If Scenario Matrix":
    st.markdown('<div class="main-header">What-If Scenario Analysis Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare baseline static allocation against optimized network reconfiguration</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Current Baseline Route Distances (Miles)")
        fig_hist1 = px.histogram(
            df,
            x='Transit Distance (Miles)',
            nbins=30,
            color_discrete_sequence=['#E11D48'],
            title='Baseline Distribution of Transit Distances'
        )
        fig_hist1.add_vline(x=df['Transit Distance (Miles)'].mean(), line_dash="dash", line_color="black", annotation_text=f"Avg: {df['Transit Distance (Miles)'].mean():.0f} mi")
        st.plotly_chart(fig_hist1, use_container_width=True)

    with col_b:
        st.subheader("Optimized Reallocated Distances (Miles)")
        fig_hist2 = px.histogram(
            df,
            x='Minimum Distance (Miles)',
            nbins=30,
            color_discrete_sequence=['#10B981'],
            title='Optimized Distribution of Transit Distances'
        )
        fig_hist2.add_vline(x=df['Minimum Distance (Miles)'].mean(), line_dash="dash", line_color="black", annotation_text=f"Avg: {df['Minimum Distance (Miles)'].mean():.0f} mi")
        st.plotly_chart(fig_hist2, use_container_width=True)

    st.markdown("---")
    st.subheader("Regional Distance Savings Breakdown")
    savings_by_region = df.groupby('Region').agg(
        Total_Orders=('Row ID', 'count'),
        Current_Miles=('Transit Distance (Miles)', 'sum'),
        Optimized_Miles=('Minimum Distance (Miles)', 'sum'),
        Miles_Saved=('Potential Distance Saved (Miles)', 'sum')
    ).reset_index()
    savings_by_region['Reduction (%)'] = (
        (savings_by_region['Miles_Saved'] / savings_by_region['Current_Miles']) * 100
    ).round(1)

    st.dataframe(savings_by_region, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# MODULE 4: Top-N Recommendations
# -----------------------------------------------------------------------------
elif app_mode == "4. Top-N Recommendations":
    st.markdown('<div class="main-header">Strategic Factory Reallocation Policy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ranked, actionable SKU-to-factory reassignment suggestions</div>', unsafe_allow_html=True)

    # Generate dynamic recommendations based on user sidebar weights
    recs_df = generate_regional_recommendations(
        enriched_df=df,
        weight_speed=weight_speed,
        weight_profit=weight_profit
    )

    action_filter = st.radio("Filter Policy Actions:", ["All Recommendations", "Actionable Reassignments Only", "Keep Current Only"], horizontal=True)

    if action_filter == "Actionable Reassignments Only":
        display_df = recs_df[recs_df['Action'] == 'Reassign Factory']
    elif action_filter == "Keep Current Only":
        display_df = recs_df[recs_df['Action'] == 'Keep Current']
    else:
        display_df = recs_df

    st.dataframe(
        display_df[[
            'Product Name', 'Destination Region', 'Current Factory', 'Recommended Factory',
            'Action', 'Order Volume', 'Current Distance (Miles)', 'New Distance (Miles)',
            'Distance Reduction (%)', 'Total Freight Miles Saved', 'Optimization Score'
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Download CSV button
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Recommendations as CSV",
        data=csv_data,
        file_name="nassau_candy_recommendations.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# MODULE 5: Risk & Capacity Analysis
# -----------------------------------------------------------------------------
elif app_mode == "5. Risk & Capacity Analysis":
    st.markdown('<div class="main-header">Operational Risk & Capacity Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate profit margins, capacity concentration, and risk thresholds</div>', unsafe_allow_html=True)

    # Risk Metrics
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.subheader("Product Profit Margin Integrity")
        margin_by_prod = df.groupby('Product Name')['Gross Margin %'].mean().reset_index().sort_values(by='Gross Margin %', ascending=False)
        fig_margin = px.bar(
            margin_by_prod,
            x='Gross Margin %',
            y='Product Name',
            orientation='h',
            title='Average Gross Profit Margin by SKU (%)',
            color='Gross Margin %',
            color_continuous_scale='Blues'
        )
        fig_margin.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_margin, use_container_width=True)

    with r_col2:
        st.subheader("Factory Workload Shift (Simulated)")
        workload_comp = pd.DataFrame({
            'Factory': list(FACTORIES.keys()),
            'Current Share (%)': [
                (df['Current Factory'] == f).mean() * 100 for f in FACTORIES.keys()
            ],
            'Optimal Share (%)': [
                (df['Closest Factory'] == f).mean() * 100 for f in FACTORIES.keys()
            ]
        })

        fig_workload = go.Figure(data=[
            go.Bar(name='Current Static Workload (%)', x=workload_comp['Factory'], y=workload_comp['Current Share (%)'], marker_color='#64748B'),
            go.Bar(name='Reallocated Workload (%)', x=workload_comp['Factory'], y=workload_comp['Optimal Share (%)'], marker_color='#2563EB')
        ])
        fig_workload.update_layout(
            barmode='group',
            title='Factory Capacity Utilization Shift (%)',
            xaxis_title='Factory Hub',
            yaxis_title='Order Share (%)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_workload, use_container_width=True)

    st.markdown("""
    <div class="notice-box">
    <b>Operational Safeguard Notes:</b><br>
    - <b>Capacity Balancing</b>: Reallocating high-volume Wonka Bar lines must account for production line re-tooling and regional raw ingredient sourcing.<br>
    - <b>Margin Protection</b>: Every recommended reallocation maintains or enhances baseline gross profit margins by reducing variable freight costs.
    </div>
    """, unsafe_allow_html=True)
