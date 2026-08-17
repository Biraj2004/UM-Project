"""
Predictive Modeling Engine for Nassau Candy
--------------------------------------------
This module trains and evaluates multiple Machine Learning regression models
to predict shipping lead times based on product, factory, route distance,
customer region, and shipping mode.

Models trained:
1. Linear Regression (Baseline)
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

The best performing model is serialized to `models/lead_time_model.pkl`.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_dataset(file_path="data/processed/nassau_candy_enriched.csv"):
    """
    Loads enriched dataset for training.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Enriched data not found at {file_path}. Run src/data_pipeline.py first.")
    return pd.read_csv(file_path)


def prepare_features(df):
    """
    Separates feature matrix (X) and target variable (y).
    """
    # Target variable is Lead Time in days
    y = df['Lead Time (Days)']

    # Categorical and numerical feature columns
    categorical_cols = [
        'Product Name',
        'Division',
        'Current Factory',
        'Region',
        'State/Province',
        'Ship Mode'
    ]

    numerical_cols = [
        'Transit Distance (Miles)',
        'Units',
        'Sales',
        'Cost',
        'Order Month'
    ]

    X = df[categorical_cols + numerical_cols].copy()
    return X, y, categorical_cols, numerical_cols


def build_preprocessor(categorical_cols, numerical_cols):
    """
    Creates a scikit-learn ColumnTransformer that one-hot encodes
    categorical variables and scales numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
    return preprocessor


def train_and_evaluate_models(X, y, categorical_cols, numerical_cols, random_state=42):
    """
    Trains multiple regression models, benchmarks their performance
    using Train/Test split and 5-Fold Cross Validation.
    """
    # 80/20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    # Candidate models to evaluate
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    }

    results = []
    trained_pipelines = {}

    print(f"Training and evaluating {len(models)} candidate models...\n")

    for model_name, model_estimator in models.items():
        # Build full pipeline: Preprocessing -> Regressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_estimator)
        ])

        # Fit model on training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        # Predictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Performance Metrics on Test Set
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 5-Fold Cross Validation (R2 Score)
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        cv_r2_mean = cv_scores.mean()

        results.append({
            "Model": model_name,
            "Test MAE (Days)": round(test_mae, 2),
            "Test RMSE (Days)": round(test_rmse, 2),
            "Test R2": round(test_r2, 4),
            "5-Fold CV R2": round(cv_r2_mean, 4)
        })

    results_df = pd.DataFrame(results).sort_values(by="Test R2", ascending=False)
    print("--- Model Benchmark Results ---")
    print(results_df.to_string(index=False))

    # Identify the best model based on highest Test R2 / lowest RMSE
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\nBest Model Selected: {best_model_name}")

    return results_df, best_pipeline, best_model_name


def save_model(pipeline, model_path="models/lead_time_model.pkl"):
    """
    Serializes the trained pipeline (preprocessor + model) to disk.
    """
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(pipeline, model_path)
    print(f"Trained model pipeline successfully saved to: {model_path}")


def load_trained_model(model_path="models/lead_time_model.pkl"):
    """
    Loads the trained pipeline from disk for predictions.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run src/model_engine.py first.")
    return joblib.load(model_path)


def run_training_pipeline():
    """
    Executes the full model training and serialization pipeline.
    """
    df = load_dataset()
    X, y, categorical_cols, numerical_cols = prepare_features(df)
    results_df, best_pipeline, best_name = train_and_evaluate_models(
        X, y, categorical_cols, numerical_cols
    )
    save_model(best_pipeline)

    # Save benchmark results table for reports and dashboard
    results_path = "data/processed/model_benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Benchmark results table saved to: {results_path}")

    return best_pipeline, results_df


if __name__ == "__main__":
    run_training_pipeline()
