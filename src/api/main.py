"""
FastAPI Server for PathSnap Two-Laptop Deployment (Laptop 1)
-----------------------------------------------------------
Hosts the trained machine learning models, HMM Viterbi decoder,
and handles server-side evaluation with isolated ground-truth labels.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

from src.data.road_network import load_road_network, load_route
from src.data.trajectory_loader import load_track
from src.pipeline.features import generate_features, FEATURE_COLUMNS
from src.hmm.transitions import RoadNetworkTransitionModel
from src.hmm.emissions import SpatialMLEmissionModel
from src.hmm.viterbi import viterbi_decode, compute_route_continuity

app = FastAPI(title="PathSnap Map-Matching API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_dir() -> Path:
    """Dataset location: $PATHSNAP_DATA_DIR, else common local folders."""
    candidates = [
        os.environ.get("PATHSNAP_DATA_DIR"),
        PROJECT_ROOT / "data" / "Student folder",
        PROJECT_ROOT / "data",
        "C:/Users/aarun/Documents/clg/sem5/Machine Learning/hackathon/data_extracted/Student folder",
    ]
    for c in candidates:
        if c and (Path(c) / "student_test").exists():
            return Path(c)
    return Path(candidates[0] or candidates[1])


DATA_DIR = _resolve_data_dir()
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_tuned_model.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "artifacts" / "random_forest_model.joblib"

# Global loaded artifacts cache
CACHED_MODEL = None
if MODEL_PATH.exists():
    try:
        data = joblib.load(MODEL_PATH)
        CACHED_MODEL = data.get("model", data) if isinstance(data, dict) else data
        print(f"Loaded model artifact from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: could not load model: {e}")


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Distributed Route Reconstruction API",
        "laptop": "Laptop 1 (Model Server)",
        "model_loaded": CACHED_MODEL is not None,
    }


@app.get("/trips")
def list_available_trips():
    """Lists available student testing trips for evaluator selection."""
    test_dir = DATA_DIR / "student_test"
    if not test_dir.exists():
        test_dir = DATA_DIR / "train"

    trip_ids = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    return {"available_trips": trip_ids, "count": len(trip_ids)}


class EvaluationResponse(BaseModel):
    trip_id: str
    total_observations: int
    top1_accuracy_percent: float
    classifier_top1_percent: float = 0.0
    route_continuity_percent: float
    candidate_miss_rate_percent: float
    inference_time_seconds: float
    reconstructed_arc_count: int
    reconstructed_route_sample: list[int]
    gps_coordinates: list[list[float]] = []
    route_coordinates: list[list[float]] = []


@app.post("/evaluate_trip/{trip_id}", response_model=EvaluationResponse)
def evaluate_trip(trip_id: str):
    """
    Processes the selected evaluation trip through the full pipeline:
    1. Spatial candidate extraction
    2. ML model candidate scoring
    3. HMM Viterbi sequential trajectory decoding
    4. Evaluates accuracy using server-side reference route.
    """
    t0 = time.perf_counter()

    # Search in student_test or train
    trip_dir = DATA_DIR / "student_test" / trip_id
    if not trip_dir.exists():
        trip_dir = DATA_DIR / "train" / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found.")

    # Load data
    network = load_road_network(trip_dir, trip_id)
    track_df = load_track(trip_dir / f"{trip_id}.track", trip_id)

    # Server-side ground truth route
    route_file = trip_dir / f"{trip_id}.route"
    gt_route = load_route(route_file) if route_file.exists() else None

    # 1. Feature generation
    feats_df, diag = generate_features(track_df, network, route=gt_route, return_diagnostics=True)

    # 2. ML Inference
    X = feats_df[FEATURE_COLUMNS].values
    if CACHED_MODEL is not None:
        if hasattr(CACHED_MODEL, "predict_proba"):
            probs = CACHED_MODEL.predict_proba(X)[:, 1]
        else:
            probs = CACHED_MODEL.predict(X).astype(float)
    else:
        # Distance-based heuristic fallback if no model trained yet
        probs = 1.0 / (1.0 + feats_df["dist_perp_m"].values)

    feats_df["ml_prob"] = probs

    # 3. HMM Sequential Decoding
    obs_candidates = []
    for obs_idx, group in feats_df.groupby("observation_idx"):
        obs_candidates.append(group.to_dict("records"))

    trans_model = RoadNetworkTransitionModel(network)
    emiss_model = SpatialMLEmissionModel()
    gps_distances = track_df["distance_m"].tolist()

    reconstructed_arcs = viterbi_decode(obs_candidates, trans_model, emiss_model, gps_distances)

    # 4. Metrics
    continuity = compute_route_continuity(reconstructed_arcs, network)

    # Reconstructed trajectory accuracy & classifier Top-1 comparison
    pipeline_acc = 0.0
    clf_top1_acc = 0.0
    if "is_true_match" in feats_df:
        # Classifier-only Top-1
        top_indices = feats_df.groupby("observation_idx")["ml_prob"].idxmax()
        clf_top1_acc = float(feats_df.loc[top_indices, "is_true_match"].mean())

        # Final map-matched reconstructed route accuracy
        obs_true_arcs = feats_df[feats_df["is_true_match"] == 1].set_index("observation_idx")["arc_id"].to_dict()
        correct_count = sum(1 for t, arc in enumerate(reconstructed_arcs) if obs_true_arcs.get(t) == arc)
        pipeline_acc = (correct_count / len(reconstructed_arcs)) if len(reconstructed_arcs) > 0 else 0.0

    # Prepare coordinates for map visualization
    gps_coords = [[float(row["lat"]), float(row["lon"])] for _, row in track_df.iterrows()]
    route_coords = []
    for arc_id in reconstructed_arcs:
        seg = network.segments[arc_id]
        route_coords.append([float(network.nodes_lat[seg.from_node]), float(network.nodes_lon[seg.from_node])])
        route_coords.append([float(network.nodes_lat[seg.to_node]), float(network.nodes_lon[seg.to_node])])

    elapsed_s = time.perf_counter() - t0

    print(f"\n[API LOG] Evaluated Trip {trip_id}:")
    print(f"  Obs: {len(track_df)}, Pipeline Acc: {pipeline_acc*100:.2f}%, Classifier Top-1: {clf_top1_acc*100:.2f}%, Continuity: {continuity*100:.2f}%, Time: {elapsed_s:.2f}s")

    return {
        "trip_id": trip_id,
        "total_observations": len(track_df),
        "top1_accuracy_percent": round(pipeline_acc * 100, 2),
        "classifier_top1_percent": round(clf_top1_acc * 100, 2),
        "route_continuity_percent": round(continuity * 100, 2),
        "candidate_miss_rate_percent": round(diag.get("candidate_miss_rate", 0.0) * 100, 2),
        "inference_time_seconds": round(elapsed_s, 2),
        "reconstructed_arc_count": len(reconstructed_arcs),
        "reconstructed_route_sample": reconstructed_arcs[:30],
        "gps_coordinates": gps_coords,
        "route_coordinates": route_coords,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
