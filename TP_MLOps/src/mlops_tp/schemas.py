#Author: Mènéli Herve Adjole
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List , Optional
from pathlib import Path
import json
import os

# =========================
# Request
# =========================
class PredictionRequest(BaseModel):
    features: List[Dict[str, Any]]

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

# =========================
# Response
# =========================
class PredictionResponse(BaseModel):
    predictions: List[int]
    task: str = "classification"
    proba: Optional[List[Dict[str, float]]] = None
    model_version: str = "0.1.0"
    latency_ms: float

    model_config = ConfigDict(
        protected_namespaces=()
    )

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# =========================
# JSON loader sécurisé
# =========================
def load_json(filename: str):
    file_path = ARTIFACTS_DIR / filename
    if not file_path.exists():
        return {}
    with open(file_path) as f:
        return json.load(f)

# =========================
# Artifacts
# =========================
_feature_schema = load_json("feature_schema.json")
_run_info = load_json("run_info.json")
_metrics = load_json("metrics.json")

# =========================
# Access functions
# =========================
def get_feature_schema():
    return _feature_schema

def get_run_info():
    return _run_info

def get_metrics():
    return _metrics

def get_task_type():
    if "task" in _run_info:
        return _run_info["task"]
    return "classification"

# =========================
# API headers (pour éviter 403 sur Render)
# =========================
def get_api_headers():
    """
    Retourne les headers à envoyer pour l'API Render.
    Ajoute User-Agent générique et éventuellement une clé API si définie.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json"
    }

    api_key = os.environ.get("RENDER_API_KEY")  # définir dans les secrets Render si nécessaire
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers
