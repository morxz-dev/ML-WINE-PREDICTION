from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
#from mlops_tp.schemas import PredictionRequest, PredictionResponse, get_feature_schema, get_task_type
from .schemas import PredictionRequest, PredictionResponse, get_feature_schema, get_task_type


app = FastAPI(title="Wine Classifier API", version="0.1.0")

# ✅ CHARGÉ UNE SEULE FOIS au démarrage !
model_path = Path("src/mlops_tp/artifacts/model.joblib")
model = joblib.load(model_path)
feature_schema = get_feature_schema()

@app.get("/health")
async def health_check():
    """Vérifie que l'API est vivante"""
    return {"status": "healthy", "model_loaded": True}

@app.get("/metadata")
async def get_metadata():
    """Renvoie infos modèle + features attendues"""
    return {
        "model_version": "0.1.0",
        "task": get_task_type(),
        "n_features": len(feature_schema),
        "features": list(feature_schema.keys()),
        "n_classes": 3
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prédiction sur nouvelles données"""
    start_time = time.time()
    
    # 1. Validation des features
    if set(request.features.keys()) != set(feature_schema.keys()):
        raise HTTPException(
            status_code=422,
            detail=f"13 features obligatoires: {list(feature_schema.keys())}"
        )
    
    # 2. Convertir en numpy array
    X = np.array(list(request.features.values())).reshape(1, -1)
    
    # 3. Prédiction
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    
    # 4. Formater réponse
    latency_ms = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        prediction=int(prediction),
        proba={str(i): float(p) for i, p in enumerate(proba)},
        latency_ms=round(latency_ms, 2)
    )
