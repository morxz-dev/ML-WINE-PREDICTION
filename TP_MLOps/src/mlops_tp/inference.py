#Author: Mènéli Herve Adjole
import pytest
import joblib
from pathlib import Path
import numpy as np


def get_model_path():
    """Construit un chemin robuste vers le modèle"""
    return Path(__file__).resolve().parents[1] / "src" / "mlops_tp" / "artifacts" / "model.joblib"


def test_model_predict():
    """Teste que predict renvoie une classe valide (0,1,2)"""

    model_path = get_model_path()

    # Vérifie que le modèle existe
    assert model_path.exists(), f"Modèle introuvable: {model_path}"

    model = joblib.load(model_path)

    # Seed pour reproductibilité
    np.random.seed(42)

    # Données de test (13 features Wine)
    X_test = np.random.randn(1, 13).astype(np.float32)

    # Test predict()
    prediction = model.predict(X_test)[0]

    assert prediction in [0, 1, 2], f"Prediction invalide: {prediction}"

    # Test predict_proba() si disponible
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[0]

        assert len(proba) == 3, "3 probas attendues pour 3 classes"
        assert np.all(proba >= 0) and np.all(proba <= 1), "Probas entre 0 et 1"
        assert np.isclose(proba.sum(), 1.0), "Probas doivent sommer à 1"


#Author: Mènéli Herve Adjole