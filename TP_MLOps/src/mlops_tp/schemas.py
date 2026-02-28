from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
from pathlib import Path
import json
import os

# features accepte TOUS les types (float, str, int...)
class PredictionRequest(BaseModel):
    features: Dict[str, Any]  # ← CHANGÉ : Any au lieu de float
    
    model_config = ConfigDict(
        extra='ignore',        # Ignore clés inconnues
        validate_assignment=True
    )

class PredictionResponse(BaseModel):
    prediction: int
    task: str = "classification"
    proba: Dict[str, float]  # ← proba au lieu de features
    model_version: str = "0.1.0"
    latency_ms: float
    
    model_config = ConfigDict(protected_namespaces=())

# ✅ Chemins absolus (robuste)
_artifacts_dir = Path(__file__).parent / "artifacts"
_feature_schema = json.loads((_artifacts_dir / "feature_schema.json").read_text())
_run_info = json.loads((_artifacts_dir / "run_info.json").read_text())

def get_feature_schema() -> Dict[str, Any]:
    return _feature_schema

def get_task_type() -> str:
    return "classification"
