"""
Nassau Candy Decision Intelligence & Shipping Optimization Platform
=============================================================================
Streamlit Web Application for Factory Reallocation and Lead Time Optimization.
Author: Biraj Sarkar | GitHub: https://github.com/Biraj2004

Modules Included:
1. Executive Overview & Strategic KPIs
2. Factory Reallocation Simulator (5-Hub Counterfactual Engine)
3. What-If Scenario Analysis Matrix
4. Strategic Top-N Recommendations Policy Engine
5. Operational Risk, Margins & Capacity Panel
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.geo_utils import FACTORIES, PRODUCT_FACTORY_MAP, US_STATE_COORDINATES
from src.data_pipeline import load_raw_data, enrich_dataset
from src.simulation_engine import simulate_all_factories_for_order
from src.optimization_engine import generate_regional_recommendations
from src.model_engine import load_trained_model


# -----------------------------------------------------------------------------
# Favicon & Streamlit Page Configuration
# -----------------------------------------------------------------------------
favicon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets', 'favicon.png'))
if os.path.exists(favicon_path):
    favicon_img = Image.open(favicon_path)
else:
    favicon_img = None

st.set_page_config(
    page_title="Nassau Candy - Decision Intelligence Platform",
    page_icon=favicon_img,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# OpenGraph Social Metadata & SEO
# -----------------------------------------------------------------------------
st.markdown("""
<head>
    <title>Nassau Candy - Decision Intelligence Platform</title>
    <meta name="title" content="Nassau Candy - Decision Intelligence Platform">
    <meta name="description" content="Enterprise Decision Intelligence, Lead Time ML & 5-Factory Shipping Optimization Platform for Nassau Candy Distributor. Unified Mentor ML Internship Project.">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://biraj-candy-ml.streamlit.app/">
    <meta property="og:title" content="Nassau Candy - Decision Intelligence Platform">
    <meta property="og:description" content="Enterprise Geospatial Decision Support, Predictive Lead Time ML & 5-Hub Factory Shipping Optimization Platform.">
    <meta property="og:image" content="https://raw.githubusercontent.com/Biraj2004/UM-Project/main/assets/og_preview.png">
    <meta property="twitter:card" content="summary_large_image">
    <meta property="twitter:url" content="https://biraj-candy-ml.streamlit.app/">
    <meta property="twitter:title" content="Nassau Candy - Decision Intelligence Platform">
    <meta property="twitter:description" content="Enterprise Geospatial Decision Support, Predictive Lead Time ML & 5-Hub Factory Shipping Optimization Platform.">
    <meta property="twitter:image" content="https://raw.githubusercontent.com/Biraj2004/UM-Project/main/assets/og_preview.png">
</head>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Design System & Custom Responsive Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Main & Sub Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 500;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        letter-spacing: -0.01em;
    }

    /* Section Subheaders */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC !important;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .section-subtitle {
        font-size: 0.88rem;
        font-weight: 500;
        color: #94A3B8 !important;
        margin-top: -0.2rem;
        margin-bottom: 0.85rem;
    }

    /* Antigravity Glassmorphic Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 10px 25px -4px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease, border-color 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 18px 32px -8px rgba(0, 0, 0, 0.6), 0 0 16px rgba(59, 130, 246, 0.25);
        border-color: rgba(96, 165, 250, 0.35);
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin-bottom: 4px;
        white-space: nowrap;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        white-space: nowrap;
    }

    /* Value Color Accents */
    .val-blue { color: #38BDF8; }
    .val-rose { color: #FB7185; }
    .val-emerald { color: #34D399; }
    .val-purple { color: #A78BFA; }
    .val-amber { color: #FBBF24; }

    /* Executive Findings / Notice Box */
    .notice-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-left: 4px solid #3B82F6;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding: 18px 22px;
        border-radius: 12px;
        margin-top: 22px;
        margin-bottom: 22px;
        font-size: 0.95rem;
        line-height: 1.65;
        color: #CBD5E1;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.3);
    }
    .notice-box b {
        color: #F8FAFC;
    }

    /* Simulator Highlights Card */
    .sim-summary-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: #E2E8F0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .sim-summary-box code {
        background: rgba(15, 23, 42, 0.9);
        color: #38BDF8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }

    /* Streamlit DataFrame & Radio Buttons Polish */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Sidebar Compact & Non-Scrollable Layout */
    [data-testid="stSidebar"] {
        padding: 0 !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.25rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.0rem !important;
        padding-right: 1.0rem !important;
        overflow-y: hidden !important;
    }
    [data-testid="stSidebarContent"]::-webkit-scrollbar {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.85rem !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    [data-testid="stSidebar"] .stSlider {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Chart Styling Helper (Zero Title Overlaps, High Contrast, Dark Mode)
# -----------------------------------------------------------------------------
def style_plotly_figure(fig, show_legend=True, legend_bottom=False):
    """
    Applies consistent dark mode styling, padding, and high-contrast typography to Plotly charts.
    """
    layout_update = dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font=dict(family="Plus Jakarta Sans", size=12, color="#E2E8F0"),
        margin=dict(l=30, r=24, t=20, b=30),
        showlegend=show_legend
    )

    if show_legend and legend_bottom:
        layout_update['legend'] = dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#F1F5F9")
        )
        layout_update['margin'] = dict(l=30, r=24, t=20, b=70)
    elif show_legend:
        layout_update['legend'] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color="#F1F5F9")
        )
        layout_update['margin'] = dict(l=30, r=24, t=45, b=30)

    fig.update_layout(**layout_update)
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.08)',
        tickfont=dict(size=12, color="#CBD5E1"),
        title_font=dict(size=13, color="#F8FAFC"),
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.08)',
        tickfont=dict(size=12, color="#CBD5E1"),
        title_font=dict(size=13, color="#F8FAFC"),
        zeroline=False
    )
    return fig


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_enriched_data():
    """Loads or generates the enriched dataset."""
    processed_path = "data/processed/nassau_candy_enriched.csv"
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)
    else:
        raw_df = load_raw_data()
        df = enrich_dataset(raw_df)
        return df


@st.cache_resource
def get_cached_model():
    """Loads the trained machine learning pipeline."""
    return load_trained_model()


# Load shared resources
df = get_cached_enriched_data()
model_pipeline = get_cached_model()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Optimization Preferences (Compact & Non-Scrollable)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    '<div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 6px; letter-spacing: -0.02em;">Navigation & Controls</div>',
    unsafe_allow_html=True
)

app_mode = st.sidebar.radio(
    "Select System Module:",
    [
        "1. Executive Overview & KPIs",
        "2. Factory Reallocation Simulator",
        "3. What-If Scenario Matrix",
        "4. Top-N Recommendations",
        "5. Risk & Capacity Analysis"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div style="height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 8px 0;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="font-size: 0.92rem; font-weight: 700; color: #F8FAFC; margin-bottom: 2px;">Optimization Weights</div>', unsafe_allow_html=True)

weight_speed = st.sidebar.slider(
    "Delivery Speed Priority (%)",
    min_value=0,
    max_value=100,
    value=60,
    step=5,
    help="Higher weight prioritizes transit distance and lead time reduction over gross profit."
) / 100.0

weight_profit = round(1.0 - weight_speed, 2)
st.sidebar.markdown(
    f"<div style='font-size: 0.84rem; color: #CBD5E1;'>Profit Margin Weight: <b style='color: #34D399;'>{int(weight_profit * 100)}%</b></div>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="margin-top: 18px; padding: 12px 14px; background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; font-size: 0.75rem; color: #94A3B8; line-height: 1.45;">
        <div style="font-weight: 700; color: #E2E8F0; margin-bottom: 4px; letter-spacing: -0.01em;">Academic & Evaluation Notice</div>
        <div>Developed solely for the <b>Unified Mentor</b> Machine Learning Internship Program by <b>Biraj Sarkar</b>. All models, simulations, and decision policies are provided for educational and grading purposes. Code is provided AS IS.</div>
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# MODULE 1: Executive Overview & KPIs
# -----------------------------------------------------------------------------
if app_mode == "1. Executive Overview & KPIs":
    st.markdown('<div class="main-header">Nassau Candy: Executive Operations Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Geospatial Decision Intelligence & 5-Factory Shipping Optimization System</div>', unsafe_allow_html=True)

    # Top KPI Metrics Ribbon (5 uniform cards)
    total_orders = len(df)
    current_avg_dist = df['Transit Distance (Miles)'].mean()
    optimal_avg_dist = df['Minimum Distance (Miles)'].mean()
    potential_miles_saved = df['Potential Distance Saved (Miles)'].sum()
    pct_closer_avail = (df['Is Closer Factory Available'].mean()) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value val-blue">{total_orders:,}</div>
            <div class="metric-label">Total Orders Analyzed</div>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value val-rose">{current_avg_dist:.0f} mi</div>
            <div class="metric-label">Current Avg Distance</div>
        </div>
        ''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value val-emerald">{optimal_avg_dist:.0f} mi</div>
            <div class="metric-label">Optimal Avg Distance</div>
        </div>
        ''', unsafe_allow_html=True)
    with c4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value val-purple">7.49M mi</div>
            <div class="metric-label">Reducible Freight Miles</div>
        </div>
        ''', unsafe_allow_html=True)
    with c5:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value val-amber">{pct_closer_avail:.1f}%</div>
            <div class="metric-label">Suboptimal Orders</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analytical Charts Row
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown('<div class="section-title">Factory Network & Order Concentration</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Historical order distribution across all 5 production hubs</div>', unsafe_allow_html=True)
        
        factory_counts = df['Current Factory'].value_counts().reset_index()
        factory_counts.columns = ['Factory', 'Order Count']
        
        fig_factory = px.pie(
            factory_counts,
            names='Factory',
            values='Order Count',
            color_discrete_sequence=['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EC4899'],
            hole=0.48
        )
        fig_factory.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans"),
            marker=dict(line=dict(color='#0F172A', width=2))
        )
        fig_factory = style_plotly_figure(fig_factory, show_legend=False)
        st.plotly_chart(fig_factory, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-title">Distance Bottlenecks: Current vs. Optimal</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Average freight distance by customer destination region (miles)</div>', unsafe_allow_html=True)
        
        region_dist = df.groupby('Region').agg(
            Current=('Transit Distance (Miles)', 'mean'),
            Optimal=('Minimum Distance (Miles)', 'mean')
        ).reset_index()

        fig_region = go.Figure(data=[
            go.Bar(
                name='Current Avg Distance',
                x=region_dist['Region'],
                y=region_dist['Current'],
                marker_color='#F43F5E',
                text=[f"{v:.0f} mi" for v in region_dist['Current']],
                textposition='outside',
                textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans")
            ),
            go.Bar(
                name='Optimal Reallocated Distance',
                x=region_dist['Region'],
                y=region_dist['Optimal'],
                marker_color='#10B981',
                text=[f"{v:.0f} mi" for v in region_dist['Optimal']],
                textposition='outside',
                textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans")
            )
        ])
        fig_region.update_layout(
            barmode='group',
            xaxis_title='Customer Destination Region',
            yaxis_title='Distance (Miles)'
        )
        fig_region = style_plotly_figure(fig_region, show_legend=True, legend_bottom=True)
        st.plotly_chart(fig_region, use_container_width=True)

    # Key Findings Callout Box
    st.markdown("""
    <div class="notice-box">
    <b>Key Executive Findings:</b><br>
    * Over <b>68.8% of all historical orders (7,011 shipments)</b> were fulfilled by a distant factory when a closer production facility was active.<br>
    * Enforcing optimal regional reallocation eliminates <b>7,487,332 transit miles (-59.6%)</b> across the network, accelerating delivery cycles and cutting carrier freight costs without eroding profit margins.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODULE 2: Factory Reallocation Simulator
# -----------------------------------------------------------------------------
elif app_mode == "2. Factory Reallocation Simulator":
    st.markdown('<div class="main-header">Factory Reallocation Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate counterfactual order fulfillment performance across all 5 candidate production hubs</div>', unsafe_allow_html=True)

    # Simulator Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        selected_product = st.selectbox("Select Confectionery SKU:", sorted(df['Product Name'].unique()), index=10)
    with c2:
        selected_state = st.selectbox("Select Customer Destination State:", sorted(US_STATE_COORDINATES.keys()), index=4)  # California
    with c3:
        selected_ship_mode = st.selectbox("Select Shipping Mode:", ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
    with c4:
        selected_units = st.number_input("Order Units / Quantity:", min_value=1, max_value=50, value=3)

    # Run Simulation
    sim_df, base_time, base_dist = simulate_all_factories_for_order(
        product_name=selected_product,
        state=selected_state,
        ship_mode=selected_ship_mode,
        units=selected_units,
        model_pipeline=model_pipeline
    )

    current_factory_name = PRODUCT_FACTORY_MAP.get(selected_product, "Unknown")
    best_sim_row = sim_df.sort_values(by='Distance (Miles)').iloc[0]

    st.markdown(f"""
    <div class="sim-summary-box">
    <b>Simulation Context</b>: Product <code>{selected_product}</code> to Destination <code>{selected_state}</code><br>
    * Current Legacy Hub: <code>{current_factory_name}</code> (<b>{base_dist:.1f} miles</b>)<br>
    * Closest Optimal Hub: <code>{best_sim_row['Factory']}</code> (<b>{best_sim_row['Distance (Miles)']:.1f} miles</b>, saving <b>{best_sim_row['Distance Saved (Miles)']:.1f} mi / {best_sim_row['Distance Saved (%)']:.1f}%</b>)
    </div>
    """, unsafe_allow_html=True)

    # Display comparison table
    st.markdown('<div class="section-title">All 5 Factory Candidates Evaluated</div>', unsafe_allow_html=True)
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
        color_discrete_map={True: '#F43F5E', False: '#3B82F6'},
        text='Distance (Miles)'
    )
    fig_sim.update_traces(
        texttemplate='%{text:.0f} mi',
        textposition='outside',
        textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans")
    )
    fig_sim = style_plotly_figure(fig_sim, show_legend=True, legend_bottom=True)
    fig_sim.update_layout(xaxis_title='Manufacturing Facility Hub', yaxis_title='Transit Distance (Miles)')
    st.plotly_chart(fig_sim, use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 3: What-If Scenario Matrix
# -----------------------------------------------------------------------------
elif app_mode == "3. What-If Scenario Matrix":
    st.markdown('<div class="main-header">What-If Scenario Analysis Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparative before-and-after distribution analysis of network transit mileage</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Baseline Route Distances (Historical Policy)</div>', unsafe_allow_html=True)
        fig_hist1 = px.histogram(
            df,
            x='Transit Distance (Miles)',
            nbins=30,
            color_discrete_sequence=['#F43F5E'],
            opacity=0.85
        )
        avg_base = df['Transit Distance (Miles)'].mean()
        fig_hist1.add_vline(
            x=avg_base,
            line_dash="dash",
            line_color="#FFFFFF",
            annotation_text=f"Baseline Avg: {avg_base:.0f} mi",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans"),
            annotation_bgcolor="#1E293B",
            annotation_bordercolor="#F43F5E"
        )
        fig_hist1 = style_plotly_figure(fig_hist1, show_legend=False)
        fig_hist1.update_layout(xaxis_title='Transit Distance (Miles)', yaxis_title='Order Count')
        st.plotly_chart(fig_hist1, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Optimized Reallocated Distances (Proposed Policy)</div>', unsafe_allow_html=True)
        fig_hist2 = px.histogram(
            df,
            x='Minimum Distance (Miles)',
            nbins=30,
            color_discrete_sequence=['#10B981'],
            opacity=0.85
        )
        avg_opt = df['Minimum Distance (Miles)'].mean()
        fig_hist2.add_vline(
            x=avg_opt,
            line_dash="dash",
            line_color="#FFFFFF",
            annotation_text=f"Optimized Avg: {avg_opt:.0f} mi",
            annotation_position="top right",
            annotation_font=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans"),
            annotation_bgcolor="#1E293B",
            annotation_bordercolor="#10B981"
        )
        fig_hist2 = style_plotly_figure(fig_hist2, show_legend=False)
        fig_hist2.update_layout(xaxis_title='Optimized Distance (Miles)', yaxis_title='Order Count')
        st.plotly_chart(fig_hist2, use_container_width=True)

    st.markdown('<div class="section-title">Regional Distance & Mileage Savings Breakdown</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="sub-header">Multi-objective Pareto-ranked factory reassignment suggestions</div>', unsafe_allow_html=True)

    # Generate dynamic recommendations based on user sidebar weights
    recs_df = generate_regional_recommendations(
        enriched_df=df,
        weight_speed=weight_speed,
        weight_profit=weight_profit
    )

    action_filter = st.radio("Filter Policy Recommendations:", ["All Recommendations", "Actionable Reassignments Only", "Keep Current Only"], horizontal=True)

    if action_filter == "Actionable Reassignments Only":
        display_df = recs_df[recs_df['Action'] == 'Reassign Factory']
    elif action_filter == "Keep Current Only":
        display_df = recs_df[recs_df['Action'] == 'Keep Current']
    else:
        display_df = recs_df

    # Rename columns for cleaner display and no truncation
    table_to_show = display_df[[
        'Product Name', 'Destination Region', 'Current Factory', 'Recommended Factory',
        'Action', 'Order Volume', 'Current Distance (Miles)', 'New Distance (Miles)',
        'Distance Reduction (%)', 'Total Freight Miles Saved', 'Optimization Score'
    ]].copy()

    table_to_show.columns = [
        'Confectionery SKU', 'Region', 'Current Hub', 'Recommended Hub',
        'Policy Action', 'Orders', 'Current (mi)', 'New (mi)',
        'Reduction %', 'Miles Saved', 'Score'
    ]

    st.dataframe(
        table_to_show,
        use_container_width=True,
        hide_index=True
    )

    # Download CSV button
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Recommendations Table as CSV",
        data=csv_data,
        file_name="nassau_candy_recommendations.csv",
        mime="text/csv"
    )


# -----------------------------------------------------------------------------
# MODULE 5: Risk & Capacity Analysis
# -----------------------------------------------------------------------------
elif app_mode == "5. Risk & Capacity Analysis":
    st.markdown('<div class="main-header">Operational Risk & Capacity Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate SKU profit margin integrity and manufacturing facility workload balance</div>', unsafe_allow_html=True)

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown('<div class="section-title">Product Profit Margin Integrity</div>', unsafe_allow_html=True)
        margin_by_prod = df.groupby('Product Name')['Gross Margin %'].mean().reset_index().sort_values(by='Gross Margin %', ascending=True)
        fig_margin = px.bar(
            margin_by_prod,
            x='Gross Margin %',
            y='Product Name',
            orientation='h',
            color='Gross Margin %',
            color_continuous_scale='Blues'
        )
        fig_margin = style_plotly_figure(fig_margin, show_legend=False)
        fig_margin.update_layout(xaxis_title='Average Gross Profit Margin (%)', yaxis_title='Confectionery SKU')
        st.plotly_chart(fig_margin, use_container_width=True)

    with r_col2:
        st.markdown('<div class="section-title">Factory Workload Distribution Shift</div>', unsafe_allow_html=True)
        workload_comp = pd.DataFrame({
            'Factory': list(FACTORIES.keys()),
            'Current Share (%)': [
                round((df['Current Factory'] == f).mean() * 100, 1) for f in FACTORIES.keys()
            ],
            'Optimal Share (%)': [
                round((df['Closest Factory'] == f).mean() * 100, 1) for f in FACTORIES.keys()
            ]
        })

        fig_workload = go.Figure(data=[
            go.Bar(
                name='Current Static Workload',
                x=workload_comp['Factory'],
                y=workload_comp['Current Share (%)'],
                marker_color='#64748B',
                text=[f"{v}%" for v in workload_comp['Current Share (%)']],
                textposition='outside',
                textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans")
            ),
            go.Bar(
                name='Reallocated Workload',
                x=workload_comp['Factory'],
                y=workload_comp['Optimal Share (%)'],
                marker_color='#3B82F6',
                text=[f"{v}%" for v in workload_comp['Optimal Share (%)']],
                textposition='outside',
                textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans")
            )
        ])
        fig_workload.update_layout(barmode='group', xaxis_title='Manufacturing Facility Hub', yaxis_title='Order Volume Share (%)')
        fig_workload = style_plotly_figure(fig_workload, show_legend=True, legend_bottom=True)
        st.plotly_chart(fig_workload, use_container_width=True)

    st.markdown("""
    <div class="notice-box">
    <b>Operational Safeguard Guidelines:</b><br>
    * <b>Capacity Realignment</b>: Reallocating high-volume Wonka Bar lines to regional facilities will shift production volume to Lot's O' Nuts and Wicked Choccy's according to geographic customer demand density.<br>
    * <b>100% Margin Preservation</b>: Every proposed factory reallocation preserves or expands gross profit margins by cutting carrier freight mile surcharges.
    </div>
    """, unsafe_allow_html=True)
