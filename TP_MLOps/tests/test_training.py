#Author: Mènéli Herve Adjole
import pytest
import os
from pathlib import Path
import joblib
from src.mlops_tp.train import artifacts_dir  

def test_training_pipeline_creates_model():
    """Teste que train.py génère model.joblib"""
    
    # 1. Lancer l'entraînement
    import subprocess
    subprocess.run(["python", "src/mlops_tp/train.py"], check=True)
    
    # 2. Vérifier que model.joblib existe
    model_path = Path("src/mlops_tp/artifacts/model.joblib")
    assert model_path.exists(), "model.joblib non généré"
    assert model_path.stat().st_size > 10000, "model.joblib trop petit"
    
    # 3. Charger et tester le modèle
    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Modèle n'a pas predict()"
    
    print("✅ test_training_pipeline_creates_model OK")

#Author: Mènéli Herve Adjole
