import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

def train_and_evaluate():
    data_path = os.path.join("data", "flash_flood_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    
    feature_cols = [
        "rainfall_intensity_mmh",
        "rainfall_24h_mm",
        "river_water_level_pct",
        "soil_moisture_pct",
        "slope_degrees",
        "vegetation_index",
        "elevation_m"
    ]
    target_col = "risk_tier"

    X = df[feature_cols]
    y = df[target_col]

    # Stratified Train-Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # Benchmarking Models: Random Forest vs Gradient Boosting
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    results = {}
    best_model_name = "Random Forest"
    best_model = None
    best_f1 = 0.0

    tier_labels = ["Low", "Moderate", "High", "Critical"]

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        
        results[name] = {
            "accuracy": round(float(acc) * 100, 2),
            "f1_macro": round(float(f1_macro) * 100, 2),
            "f1_weighted": round(float(f1_weighted) * 100, 2),
            "classification_report": classification_report(y_test, y_pred, target_names=tier_labels, output_dict=True)
        }
        
        print(f"\n--- {name} Results ---")
        print(f"Accuracy: {results[name]['accuracy']}% | F1-Score (Weighted): {results[name]['f1_weighted']}%")
        print(classification_report(y_test, y_pred, target_names=tier_labels))
        
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_model_name = name
            best_model = clf

    # Extract Feature Importances for Explainable AI
    importances = best_model.feature_importances_
    feature_importance_dict = {
        col: round(float(imp) * 100, 2)
        for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    }

    # Confusion Matrix on best model
    y_best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_best_pred).tolist()

    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "flood_model.pkl")
    joblib.dump(best_model, model_save_path)
    print(f"\n[OK] Best model ({best_model_name}) serialized to: {model_save_path}")

    # Save metadata and metrics for the frontend Streamlit dashboard
    metadata = {
        "best_model": best_model_name,
        "features": feature_cols,
        "tier_labels": tier_labels,
        "metrics": results[best_model_name],
        "feature_importances": feature_importance_dict,
        "confusion_matrix": cm,
        "model_comparison": {k: {"accuracy": v["accuracy"], "f1_weighted": v["f1_weighted"]} for k, v in results.items()}
    }

    metadata_path = os.path.join("models", "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Model metadata and evaluation metrics saved to: {metadata_path}")
    print("\nFeature Importances (Hydrological Drivers):")
    for feat, imp in feature_importance_dict.items():
        print(f" - {feat:25s}: {imp}%")

if __name__ == "__main__":
    train_and_evaluate()

