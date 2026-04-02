from fastapi import FastAPI, HTTPException
import joblib
import time
import numpy as np
from pathlib import Path
from .schemas import PredictionRequest, PredictionResponse, get_feature_schema, get_task_type

app = FastAPI(title="Wine Classifier API", version="0.1.0")

# Chemin robuste vers le modèle (compatible Docker / CI)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"

# Chargement modèle
model = joblib.load(MODEL_PATH)

# Schéma des features
feature_schema = get_feature_schema()


# Endpoint santé (obligatoire pour déploiement)
@app.get("/health")
def health():
    return {"status": "ok"}
#Author: Mènéli Herve Adjole

# Endpoint prédiction
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    # Vérification des features
    for obs in request.features:
        if set(obs.keys()) != set(feature_schema.keys()):
            raise HTTPException(
                status_code=422,
                detail=f"Chaque observation doit contenir {list(feature_schema.keys())}"
            )

    # Conversion en numpy
    X = np.array([[obs[f] for f in feature_schema.keys()] for obs in request.features])

    # Prédiction
    predictions = model.predict(X)

    # Probabilités (si dispo)
    probas = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    # Latence
    latency_ms = (time.time() - start_time) * 1000

    return PredictionResponse(
        predictions=[int(p) for p in predictions],
        proba=[
            {str(i): float(p) for i, p in enumerate(prob)}
            for prob in probas
        ] if probas is not None else None,
        model_version="0.1.0",
        task=get_task_type(),
        latency_ms=round(latency_ms, 2)
    )
#Author: Mènéli Herve Adjole