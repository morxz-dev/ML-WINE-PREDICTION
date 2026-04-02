import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature  

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from config import *

# =========================
# Dossier artefacts
# =========================
artifacts_dir = Path(__file__).parent / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("wine_classification")

with mlflow.start_run():

    # =========================
    # Chargement des données
    # =========================
    print("🍷 Chargement dataset Wine...")
    data = load_wine(as_frame=True)
    X = data.data
    y = data.target

    print(f"📊 Dataset: {X.shape[0]} échantillons, {X.shape[1]} features")
    print(f"🎯 Classes: {np.unique(y)} (distribution: {np.bincount(y)})")

    # =========================
    # Split
    # =========================
    print("\n🔪 Split train/val/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # =========================
    # Pipeline
    # =========================
    print("\n🏭 Pipeline...")
    n_estimators = 100

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # =========================
    # Entraînement
    # =========================
    print("🚀 Entraînement...")
    pipeline.fit(X_train, y_train)

    # =========================
    # Prédictions
    # =========================
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    # =========================
    # Métriques
    # =========================
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print(f"\n📈 Test: Acc={test_acc:.3f}, F1={test_f1:.3f}")

    # =========================
    # MLflow Logging
    # =========================
    # Paramètres
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", RANDOM_STATE)

    # Métriques
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    mlflow.log_metric("train_f1_macro", train_f1)
    mlflow.log_metric("val_f1_macro", val_f1)
    mlflow.log_metric("test_f1_macro", test_f1)

    # =========================
    # Artefact : matrice de confusion
    # =========================
    print("\n📊 Génération matrice de confusion...")

    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    cm_path = artifacts_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # =========================
    # Sauvegarde locale
    # =========================
    print("\n💾 Sauvegarde...")

    joblib.dump(pipeline, artifacts_dir / "model.joblib")

    metrics = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "train_f1_macro": float(train_f1),
        "val_f1_macro": float(val_f1),
        "test_f1_macro": float(test_f1),
        "hyperparams": {
            "n_estimators": n_estimators,
            "random_state": RANDOM_STATE
        }
    }

    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    feature_schema = {col: str(X[col].dtype) for col in X.columns}
    with open(artifacts_dir / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)

    run_info = {
        "dataset_name": DATASET_NAME,
        "shape": list(X.shape),
        "target": str(y.name),
        "n_classes": int(np.unique(y).size),
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "random_state": RANDOM_STATE,
        "timestamp": pd.Timestamp.now().isoformat()
    }

    with open(artifacts_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    # =========================
    # MLflow artefacts
    # =========================
    mlflow.log_artifact(artifacts_dir / "model.joblib")
    mlflow.log_artifact(artifacts_dir / "metrics.json")
    mlflow.log_artifact(artifacts_dir / "feature_schema.json")
    mlflow.log_artifact(artifacts_dir / "run_info.json")

    # MODÈLE MLflow AVEC SIGNATURE (fix warnings)
    signature = infer_signature(X_test, y_test_pred)
    input_example = X_test.iloc[[0]].to_dict('records')[0]  # Exemple input

    mlflow.sklearn.log_model(
        pipeline, 
        "wine_rf_model",  
        signature=signature,
        input_example=input_example
    )

    print("\n RUN MLflow COMPLET SANS WARNINGS !")