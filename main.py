"""
Master Entry Point: Nassau Candy Supply Chain Optimization System
=============================================================================
Author: Biraj Sarkar
GitHub: https://github.com/Biraj2004
Project: Factory Reallocation & Geospatial Shipping Decision Intelligence

Description:
------------
This is the root-level orchestrator script. Running this single file executes
the complete end-to-end pipeline chainwise:
    [Step 1] Ingests and cleans raw transactions, computing Haversine transit distances.
    [Step 2] Performs unsupervised K-Means route clustering (k=3) to isolate bottlenecks.
    [Step 3] Trains, benchmarks 5 regression algorithms, and serializes the best Random Forest.
    [Step 4] Executes multi-objective Pareto optimization and exports strategic reallocation policies.

Usage:
------
    # Run the full end-to-end pipeline
    python main.py

    # Run specific individual pipeline stages
    python main.py --step data
    python main.py --step cluster
    python main.py --step train
    python main.py --step optimize

    # Run the full pipeline and immediately launch the Streamlit web dashboard
    python main.py --app
"""

import os
import sys
import time
import argparse
import subprocess

# Ensure project root is in the Python search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_pipeline import run_pipeline as run_data_pipeline
from src.clustering import run_clustering_pipeline
from src.model_engine import run_training_pipeline
from src.optimization_engine import run_optimization_pipeline


def print_banner():
    """Prints the system header banner."""
    print("=" * 80)
    print("  NASSAU CANDY DISTRIBUTOR: GEOSPATIAL DECISION INTELLIGENCE SYSTEM")
    print("  Machine Learning, Route Clustering & Factory Reallocation Engine")
    print("  Author: Biraj Sarkar | GitHub: https://github.com/Biraj2004")
    print("=" * 80)


def step_1_data():
    """Step 1: Data Ingestion & Geospatial Distance Calculation."""
    print("\n" + "=" * 80)
    print(">> [STEP 1/4] INGESTION, DATA CLEANING & GEOSPATIAL ENRICHMENT")
    print("=" * 80)
    start = time.time()
    enriched_df = run_data_pipeline()
    elapsed = time.time() - start
    print(f">> [STEP 1 COMPLETED] Processed {len(enriched_df):,} orders in {elapsed:.2f}s.")
    return enriched_df


def step_2_cluster():
    """Step 2: Unsupervised Route Bottleneck Clustering."""
    print("\n" + "=" * 80)
    print(">> [STEP 2/4] UNSUPERVISED ROUTE BOTTLENECK CLUSTERING (K-Means, k=3)")
    print("=" * 80)
    start = time.time()
    clustered_df, summary = run_clustering_pipeline()
    elapsed = time.time() - start
    print(f">> [STEP 2 COMPLETED] Clustered {len(clustered_df):,} orders in {elapsed:.2f}s.")
    return clustered_df, summary


def step_3_train():
    """Step 3: Supervised Model Benchmarking & Random Forest Training."""
    print("\n" + "=" * 80)
    print(">> [STEP 3/4] SUPERVISED LEAD TIME REGRESSION & MODEL BENCHMARKING")
    print("=" * 80)
    start = time.time()
    best_pipeline, results_df = run_training_pipeline()
    elapsed = time.time() - start
    print(f">> [STEP 3 COMPLETED] Evaluated 5 models and saved best pipeline in {elapsed:.2f}s.")
    return best_pipeline, results_df


def step_4_optimize():
    """Step 4: Multi-Objective Pareto Optimization & Policy Generation."""
    print("\n" + "=" * 80)
    print(">> [STEP 4/4] MULTI-OBJECTIVE PARETO OPTIMIZATION & POLICY REALLOCATION")
    print("=" * 80)
    start = time.time()
    rec_df = run_optimization_pipeline()
    elapsed = time.time() - start
    print(f">> [STEP 4 COMPLETED] Generated {len(rec_df):,} policy rules in {elapsed:.2f}s.")
    return rec_df


def launch_streamlit():
    """Launches the Streamlit Web Application."""
    print("\n" + "=" * 80)
    print(">> LAUNCHING STREAMLIT WEB DASHBOARD...")
    print(">> Open your browser at: http://localhost:8501")
    print("=" * 80)
    app_path = os.path.join("app", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def run_all():
    """Runs all pipeline stages sequentially."""
    print_banner()
    total_start = time.time()

    # Step 1: Data Ingestion & Enrichment
    step_1_data()

    # Step 2: Unsupervised Clustering
    step_2_cluster()

    # Step 3: Supervised Machine Learning
    step_3_train()

    # Step 4: Multi-Objective Optimization
    step_4_optimize()

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print(">> ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f">> Total Execution Time: {total_elapsed:.2f} seconds")
    print("=" * 80)
    print("\nGenerated Artifacts & Output Files:")
    print("  [1] data/processed/nassau_candy_enriched.csv  (Enriched dataset with Haversine distances)")
    print("  [2] data/processed/nassau_candy_clustered.csv (K-Means route clustering profiles)")
    print("  [3] data/processed/model_benchmark_results.csv(5-Model regression benchmark table)")
    print("  [4] data/processed/top_recommendations.csv    (Ranked factory reallocation policies)")
    print("  [5] models/lead_time_model.pkl                (Trained Random Forest model pipeline)")
    print("\nNext Steps:")
    print("  * Launch Interactive Web Dashboard: streamlit run app/app.py")
    print("  * View Compiled Academic Report:     reports/PROJECT_REPORT.pdf")
    print("  * Inspect Jupyter Notebooks:         jupyter notebook")
    print("=" * 80)


def main():
    """CLI Argument Parser and Dispatcher."""
    parser = argparse.ArgumentParser(
        description="Nassau Candy Supply Chain Machine Learning & Decision Intelligence Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--step",
        choices=["data", "cluster", "train", "optimize", "all"],
        default="all",
        help="Specify which pipeline step to run (default: all)"
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Launch the Streamlit web dashboard after running the pipeline"
    )

    args = parser.parse_args()

    if args.step == "all":
        run_all()
    elif args.step == "data":
        print_banner()
        step_1_data()
    elif args.step == "cluster":
        print_banner()
        step_2_cluster()
    elif args.step == "train":
        print_banner()
        step_3_train()
    elif args.step == "optimize":
        print_banner()
        step_4_optimize()

    if args.app:
        launch_streamlit()


if __name__ == "__main__":
    main()
