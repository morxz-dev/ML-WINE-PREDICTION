#Author: Mènéli Herve Adjole
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List , Optional
from pathlib import Path
import json

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

from typing import Optional

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

#Author: Mènéli Herve Adjole