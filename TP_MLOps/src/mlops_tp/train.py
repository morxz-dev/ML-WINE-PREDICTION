
import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from config import *

'''artifacts_dir = Path("src/mlops_tp/artifacts")'''
artifacts_dir = Path(__file__).parent / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

#Chargement des données et affichage du dataset

print("🍷 Chargement dataset Wine...")
data = load_wine(as_frame=True)
X = data.data
y = data.target

print(f"📊 Dataset: {X.shape[0]} échantillons, {X.shape[1]} features")
print(f"🎯 Classes: {np.unique(y)} (distribution: {np.bincount(y)})")

#Separation de dataset en train/val/test
print("\n🔪 Split train/val/test...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#Pipeline de preprocessing et classification
print("\n🏭 Pipeline...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
])

#Entraînement et évaluation/metriques
print("🚀 Entraînement...")
pipeline.fit(X_train, y_train)

train_acc = accuracy_score(y_train, pipeline.predict(X_train))
val_acc = accuracy_score(y_val, pipeline.predict(X_val))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))
train_f1 = f1_score(y_train, pipeline.predict(X_train), average='macro')
val_f1 = f1_score(y_val, pipeline.predict(X_val), average='macro')
test_f1 = f1_score(y_test, pipeline.predict(X_test), average='macro')

metrics = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "train_accuracy": float(train_acc),
    "val_accuracy": float(val_acc),
    "test_accuracy": float(test_acc),
    "train_f1_macro": float(train_f1),
    "val_f1_macro": float(val_f1),
    "test_f1_macro": float(test_f1),
    "hyperparams": {"n_estimators": 100, "random_state": RANDOM_STATE}
}

print(f"\n📈 Test: Acc={test_acc:.3f}, F1={test_f1:.3f}")

#Sauvegarde  des artefacts (modèle, métriques, schema des features)
print("\n💾 Sauvegarde...")
joblib.dump(pipeline, artifacts_dir / "model.joblib")

with open(artifacts_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

feature_schema = {col: str(X[col].dtype) for col in X.columns}
with open(artifacts_dir / "feature_schema.json", "w") as f:
    json.dump(feature_schema, f, indent=2)

#Informations sur le run (dataset, shape, target, classes, split ratios, timestamp)
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

print("\n🎉 4 ARTEFACTS GÉNÉRÉS !")

