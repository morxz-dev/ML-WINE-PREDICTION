import pytest
import joblib
from pathlib import Path
import numpy as np

def test_model_predict():
    """Teste que predict renvoie une classe valide (0,1,2)"""
    
    # Charger le modèle entraîné
    model_path = Path("src/mlops_tp/artifacts/model.joblib")
    model = joblib.load(model_path)
    
    # Données de test (13 features Wine)
    X_test = np.random.randn(1, 13).astype(np.float32)
    
    # Test predict()
    prediction = model.predict(X_test)[0]
    assert prediction in [0, 1, 2], f"Prediction invalide: {prediction}"
    
    # Test predict_proba() si disponible
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)[0]
        assert len(proba) == 3, "3 probas attendues pour 3 classes"
        assert 0 <= proba.min() <= proba.max() <= 1, "Probas entre 0-1"
        assert abs(proba.sum() - 1.0) < 1e-6, "Probas somment à 1"
    
    print("✅ test_model_predict OK")
