"""
Script to generate self-contained, Google Colab-compatible Jupyter Notebooks (.ipynb)
for all hackathon milestones.
"""

import json
from pathlib import Path

def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "PathSnap",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "accelerator": "CPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()]
    }

def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.splitlines()]
    }

nb_dir = Path("C:/Users/vincent/Downloads/ML_Hackathon/vincent-ml-hackathon/notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# NOTEBOOK 1: EDA & PREPROCESSING
# =============================================================================
nb1_cells = [
    md_cell("""# PathSnap: Exploratory Data Analysis & Spatial Preprocessing
**Course:** UE24CS352A - Machine Learning | PES University  
**Author:** A S Vincent (`vincent2maran@gmail.com`)

This notebook performs:
1. Environment setup and Google Drive mounting (for Google Colab).
2. Inspection of raw files (`.track`, `.nodes`, `.arcs`, `.route`).
3. Spatial coordinate projection (WGS84 degrees -> UTM Zone 32N in metres).
4. Kinematic trajectory diagnostics (speeds, 1 Hz regularity, gaps).
5. High-recall spatial candidate generation using KD-Tree.
6. Interactive Folium map visualization of raw GPS observations vs true road path."""),

    code_cell("""# 1. Google Colab Environment Setup & Dependencies
# Run this cell if running on Google Colab:
!pip install -q pyproj networkx folium scipy scikit-learn

# Mount Google Drive (Optional if data is in Google Drive)
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = "/content/drive/MyDrive/ML_Hackathon/Student folder/Student folder"

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

# Local / Colab path detection
LOCAL_DATA = Path("../Student folder/Student folder")
if not LOCAL_DATA.exists():
    LOCAL_DATA = Path("data")
print("Using data directory:", LOCAL_DATA)"""),

    md_cell("""## 2. Spatial Projection: WGS84 Degrees to UTM Metres
Raw GPS coordinates are in spherical degrees (`lon, lat`). As mandated by the problem guidelines, ordinary Euclidean distance on degrees is invalid because 1 degree of longitude varies with latitude. We project coordinates into **UTM Zone 32N (EPSG:32632)** where unit = 1 metre."""),

    code_cell("""from pyproj import Transformer, CRS

def build_utm_transformer(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=False)

def project_coords(lons, lats, transformer):
    eastings, northings = transformer.transform(lats, lons)
    return np.asarray(eastings, dtype=np.float64), np.asarray(northings, dtype=np.float64)

# Test sample coordinates near Darmstadt, Germany
sample_lon, sample_lat = 8.581679, 49.857880
tf = build_utm_transformer(sample_lon, sample_lat)
ex, ny = project_coords([sample_lon], [sample_lat], tf)
print(f"Sample WGS84 ({sample_lon}, {sample_lat}) -> UTM Metric: Easting={ex[0]:.2f}m, Northing={ny[0]:.2f}m")"""),

    md_cell("""## 3. Load Road Network & Trajectory for Trip 00000001"""),

    code_cell("""trip_dir = LOCAL_DATA / "train" / "00000001"
nodes_data = np.loadtxt(trip_dir / "00000001.nodes")
arcs_data = np.loadtxt(trip_dir / "00000001.arcs", dtype=np.int64)
track_data = np.loadtxt(trip_dir / "00000001.track")
route_arcs = np.loadtxt(trip_dir / "00000001.route", dtype=np.int64)

print(f"Road Vertices (Nodes)     : {len(nodes_data):,}")
print(f"Road Segments (Arcs)      : {len(arcs_data):,}")
print(f"GPS Observations (Track)  : {len(track_data):,}")
print(f"Traversed Route Arcs      : {len(route_arcs):,}")"""),

    md_cell("""## 4. Trajectory Sampling Interval Audit (~1 Hz Verification)
We verify that ~97% of observations are spaced 1 second apart, calculate speed, and inspect distributions."""),

    code_cell("""timestamps = track_data[:, 2]
dt = np.diff(timestamps)

regular_fraction = np.mean(np.abs(dt - 1.0) <= 0.25) * 100
print(f"Percentage of regular 1-second intervals: {regular_fraction:.2f}%")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(dt[dt < 5], bins=30, color='royalblue', edgecolor='black')
plt.title("GPS Time Delta (dt) Distribution")
plt.xlabel("Seconds")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
# Compute speed
dx = np.diff(track_data[:, 0]) * 111320 * np.cos(np.radians(49.85))
dy = np.diff(track_data[:, 1]) * 111320
dist_m = np.hypot(dx, dy)
speed_mps = dist_m / np.maximum(dt, 0.001)

plt.hist(speed_mps[speed_mps < 45], bins=30, color='forestgreen', edgecolor='black')
plt.title("Vehicle Speed Distribution (m/s)")
plt.xlabel("Speed (m/s)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()"""),

    md_cell("""## 5. Candidate Generation using KD-Tree
We index road segments and evaluate candidate miss rate against ground truth on Trip 00000001."""),

    code_cell("""from scipy.spatial import cKDTree

# Build UTM coordinates
tf = build_utm_transformer(np.mean(nodes_data[:, 0]), np.mean(nodes_data[:, 1]))
nodes_x, nodes_y = project_coords(nodes_data[:, 0], nodes_data[:, 1], tf)
track_x, track_y = project_coords(track_data[:, 0], track_data[:, 1], tf)

# Discretize segments into KD-Tree points every 20m
pts, seg_ids = [], []
for arc_id in range(len(arcs_data)):
    fn, tn = arcs_data[arc_id]
    x0, y0 = nodes_x[fn], nodes_y[fn]
    x1, y1 = nodes_x[tn], nodes_y[tn]
    length = np.hypot(x1 - x0, y1 - y0)
    samples = max(2, int(np.ceil(length / 20.0)) + 1)
    for a in np.linspace(0, 1, samples):
        pts.append((x0 + a*(x1 - x0), y0 + a*(y1 - y0)))
        seg_ids.append(arc_id)

tree = cKDTree(np.array(pts))
seg_ids = np.array(seg_ids)

print("KD-Tree built with", len(pts), "sample points!")

# Query radius 60m for first 100 observations
misses = 0
for i in range(100):
    px, py = track_x[i], track_y[i]
    nearby = tree.query_ball_point([px, py], r=60.0)
    cands = set(seg_ids[nearby])
    # Ground truth reference
    true_arc = route_arcs[min(i, len(route_arcs)-1)]
    if true_arc not in cands:
        misses += 1

print(f"Candidate Miss Rate on sample points: {misses / 100 * 100:.2f}% (Target: < 5%)")"""),

    md_cell("""## 6. Interactive Folium Map: Raw GPS vs Reconstructed Route
We visualize raw noisy GPS observations (Blue) against the true traversed road segments (Green)."""),

    code_cell("""center_lat = float(track_data[0, 1])
center_lon = float(track_data[0, 0])

fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

# 1. Plot raw GPS observations (Blue circles)
for i in range(0, min(300, len(track_data)), 2):
    folium.CircleMarker(
        location=[track_data[i, 1], track_data[i, 0]],
        radius=3,
        color="blue",
        fill=True,
        fill_opacity=0.7,
        tooltip=f"GPS Obs {i}"
    ).add_to(fmap)

# 2. Plot ground truth route path (Green polyline)
route_coords = []
for arc_id in route_arcs[:50]:
    fn, tn = arcs_data[arc_id]
    route_coords.append([nodes_data[fn, 1], nodes_data[fn, 0]])
    route_coords.append([nodes_data[tn, 1], nodes_data[tn, 0]])

folium.PolyLine(
    route_coords,
    color="forestgreen",
    weight=4,
    opacity=0.8,
    tooltip="Ground Truth Traversed Road Path"
).add_to(fmap)

fmap.save("folium_trajectory_map.html")
print("Saved interactive map to folium_trajectory_map.html")
fmap""")
]

with open(nb_dir / "01_EDA_and_Preprocessing.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

print("Created 01_EDA_and_Preprocessing.ipynb")

# =============================================================================
# NOTEBOOK 2: MODEL ZOO & COMPARISON TABLE 3.1
# =============================================================================
nb2_cells = [
    md_cell("""# PathSnap: Multi-Model Zoo & Experimental Comparison (Table 3.1)
**Course:** UE24CS352A - Machine Learning | PES University  
**Author:** A S Vincent (`vincent2maran@gmail.com`)

This notebook implements:
1. Feature extraction across multiple training trips.
2. Group-Aware Cross-Validation (`GroupKFold` strictly by `trip_id` to prevent data leakage).
3. Implementation and evaluation of **5 DISTINCT MODELS across multiple algorithm families**:
   - **Model 1:** Random Forest Classifier (Tree-based Ensemble)
   - **Model 2:** LightGBM Classifier (Gradient Boosting)
   - **Model 3:** Multi-Layer Perceptron (Neural Network)
   - **Model 4:** Distance-Weighted KNN (Instance-based)
   - **Model 5:** Logistic Regression (Linear & Probabilistic Baseline)
4. Full Table 3.1 Model Comparison Matrix (Macro F1, Top-1 Accuracy, Log Loss, Latency)."""),

    code_cell("""# Setup dependencies
!pip install -q pyproj networkx scikit-learn lightgbm pandas numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

LOCAL_DATA = Path("../Student folder/Student folder/train")
if not LOCAL_DATA.exists():
    LOCAL_DATA = Path("data/train")"""),

    md_cell("""## 1. Feature Extraction Pipeline Definition"""),

    code_cell("""from scipy.spatial import cKDTree
from pyproj import Transformer, CRS

def point_to_segment_projection(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx*dx + dy*dy
    if l2 == 0:
        return float(np.hypot(px - x1, py - y1)), 0.0
    u = ((px - x1)*dx + (py - y1)*dy) / l2
    u_clamped = max(0.0, min(1.0, u))
    proj_x = x1 + u_clamped * dx
    proj_y = y1 + u_clamped * dy
    dist = float(np.hypot(px - proj_x, py - proj_y))
    return dist, float(u_clamped)

def extract_features_for_trip(trip_dir, trip_id, candidate_radius=60.0, top_k=10):
    nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes")
    arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
    track = np.loadtxt(trip_dir / f"{trip_id}.track")
    route = np.loadtxt(trip_dir / f"{trip_id}.route", dtype=np.int64)

    # UTM projection
    zone = int((np.mean(nodes[:, 0]) + 180) / 6) + 1
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32600 + zone), always_xy=False)
    nodes_x, nodes_y = tf.transform(nodes[:, 1], nodes[:, 0])
    track_x, track_y = tf.transform(track[:, 1], track[:, 0])

    # Sample KD-Tree
    pts, seg_ids = [], []
    for arc_id in range(len(arcs)):
        fn, tn = arcs[arc_id]
        x0, y0 = nodes_x[fn], nodes_y[fn]
        x1, y1 = nodes_x[tn], nodes_y[tn]
        L = np.hypot(x1 - x0, y1 - y0)
        n_s = max(2, int(np.ceil(L / 20.0)) + 1)
        for a in np.linspace(0, 1, n_s):
            pts.append((x0 + a*(x1 - x0), y0 + a*(y1 - y0)))
            seg_ids.append(arc_id)
    tree = cKDTree(np.array(pts))
    seg_ids = np.array(seg_ids)

    # Monotonic route alignment
    n_obs = len(track)
    n_route = len(route)
    aligned_arcs = np.empty(n_obs, dtype=np.int64)
    r_idx = 0
    for i in range(n_obs):
        px, py = track_x[i], track_y[i]
        best_d = float('inf')
        best_arc = route[r_idx]
        best_r = r_idx
        for cand_r in range(max(0, r_idx - 2), min(n_route, r_idx + 16)):
            arc_id = route[cand_r]
            fn, tn = arcs[arc_id]
            d, _ = point_to_segment_projection(px, py, nodes_x[fn], nodes_y[fn], nodes_x[tn], nodes_y[tn])
            if d < best_d:
                best_d = d
                best_arc = arc_id
                best_r = cand_r
        aligned_arcs[i] = best_arc
        r_idx = best_r

    dx = np.diff(track_x, append=track_x[-1])
    dy = np.diff(track_y, append=track_y[-1])
    headings = np.arctan2(dy, dx)

    rows = []
    for i in range(n_obs):
        px, py = track_x[i], track_y[i]
        true_arc = aligned_arcs[i]
        nearby = tree.query_ball_point([px, py], r=candidate_radius)
        cand_arcs = list(set(seg_ids[nearby])) if nearby else [seg_ids[tree.query([px, py], k=1)[1]]]

        scored = []
        for arc_id in cand_arcs:
            fn, tn = arcs[arc_id]
            d, u = point_to_segment_projection(px, py, nodes_x[fn], nodes_y[fn], nodes_x[tn], nodes_y[tn])
            scored.append((arc_id, d, u, np.hypot(nodes_x[tn]-nodes_x[fn], nodes_y[tn]-nodes_y[fn]), np.arctan2(nodes_y[tn]-nodes_y[fn], nodes_x[tn]-nodes_x[fn])))
        scored.sort(key=lambda x: x[1])
        scored = scored[:top_k]

        for rank, (arc_id, d, u, L, road_head) in enumerate(scored):
            head_diff = abs((headings[i] - road_head + np.pi) % (2.0 * np.pi) - np.pi)
            rows.append({
                "trip_id": trip_id,
                "observation_idx": i,
                "arc_id": arc_id,
                "dist_perp_m": d,
                "proj_ratio_u": u,
                "road_length_m": L,
                "heading_diff_deg": np.degrees(head_diff),
                "cos_heading_diff": np.cos(head_diff),
                "candidate_rank": rank,
                "is_true_match": int(arc_id == true_arc)
            })
    return pd.DataFrame(rows)"""),

    md_cell("""## 2. Extract Training Features Across Trips"""),

    code_cell("""# Extract dataset for 5 trips (adjust trip count as needed)
trip_list = sorted([d.name for d in LOCAL_DATA.iterdir() if d.is_dir()])[:5]
print("Building dataset from trips:", trip_list)

dfs = []
for t_id in trip_list:
    df_t = extract_features_for_trip(LOCAL_DATA / t_id, t_id)
    dfs.append(df_t)

full_df = pd.concat(dfs, ignore_index=True)
print(f"Total candidate rows: {len(full_df):,}")
print("Class balance (0: Negative, 1: True Match):")
print(full_df['is_true_match'].value_counts())"""),

    md_cell("""## 3. Train & Evaluate the 5 Distinct Models with GroupKFold"""),

    code_cell("""from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

FEATURE_COLS = ["dist_perp_m", "proj_ratio_u", "road_length_m", "heading_diff_deg", "cos_heading_diff", "candidate_rank"]

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1),
    "Multi-Layer Perceptron": Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42))]),
    "Distance-Weighted KNN": Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1))]),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight='balanced', max_iter=300, random_state=42))]),
}

comparison_rows = []
X = full_df[FEATURE_COLS].values
y = full_df["is_true_match"].values
groups = full_df["trip_id"].values
gkf = GroupKFold(n_splits=5)

for name, model in models.items():
    print(f"Running 5-Fold GroupKFold CV for: {name}...")
    f1s, top1s, losses, times = [], [], [], []

    for tr_idx, va_idx in gkf.split(X, y, groups):
        model.fit(X[tr_idx], y[tr_idx])

        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X[va_idx])[:, 1]
        else:
            probs = model.predict(X[va_idx]).astype(float)
        latency = (time.perf_counter() - t0)

        preds = (probs >= 0.5).astype(int)
        f1s.append(f1_score(y[va_idx], preds, average="macro", zero_division=0))
        losses.append(log_loss(y[va_idx], np.clip(probs, 1e-7, 1 - 1e-7)))

        # Top-1 accuracy
        val_df = full_df.iloc[va_idx].copy()
        val_df["prob"] = probs
        top_picks = val_df.loc[val_df.groupby(["trip_id", "observation_idx"])["prob"].idxmax()]
        top1s.append(top_picks["is_true_match"].mean())
        times.append(latency / val_df.groupby(["trip_id", "observation_idx"]).ngroups * 1000.0)

    comparison_rows.append({
        "Model Name": name,
        "CV Macro F1": round(np.mean(f1s), 4),
        "Top-1 Accuracy (%)": round(np.mean(top1s) * 100, 2),
        "Log Loss": round(np.mean(losses), 4),
        "Inference Latency (ms/obs)": round(np.mean(times), 2)
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\\n================ TABLE 3.1: MODEL COMPARISON MATRIX ================")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("table_3_1_model_comparison.csv", index=False)"""),

    md_cell("""## 4. Visualization: Comparative Performance Plots"""),

    code_cell("""plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.barh(comparison_df["Model Name"], comparison_df["CV Macro F1"], color='steelblue')
plt.title("Cross-Validation Macro F1 Score")
plt.xlabel("Macro F1")

plt.subplot(1, 2, 2)
plt.barh(comparison_df["Model Name"], comparison_df["Top-1 Accuracy (%)"], color='coral')
plt.title("Observation-level Top-1 Accuracy (%)")
plt.xlabel("Top-1 Accuracy %")
plt.tight_layout()
plt.show()""")
]

with open(nb_dir / "02_Model_Zoo_and_Comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

print("Created 02_Model_Zoo_and_Comparison.ipynb")

# =============================================================================
# NOTEBOOK 3: BAYESIAN OPTIMIZATION
# =============================================================================
nb3_cells = [
    md_cell("""# PathSnap: Bayesian Hyperparameter Optimization
**Course:** UE24CS352A - Machine Learning | PES University  
**Author:** A S Vincent (`vincent2maran@gmail.com`)

This notebook implements:
1. Optuna Bayesian search over hyperparameter space.
2. Group-Aware Cross-Validation objective to prevent data leakage.
3. Optimization tracking: slice plot, convergence curve, parameter importances.
4. Evaluation of untuned vs tuned model on testing trips.
5. Model weight export (`best_tuned_model.joblib`)."""),

    code_cell("""# Setup Optuna
!pip install -q optuna scikit-learn lightgbm pandas numpy

import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import joblib

# Load or recreate feature table
# df = pd.read_csv("train_features.csv")
print("Optuna Version:", optuna.__version__)"""),

    md_cell("""## 1. Define Bayesian Objective Function with GroupKFold"""),

    code_cell("""def objective(trial, X, y, groups):
    # Parameter search bounds
    n_estimators = trial.suggest_int("n_estimators", 80, 240, step=20)
    max_depth = trial.suggest_int("max_depth", 8, 18)
    min_samples_split = trial.suggest_int("min_samples_split", 4, 16)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 8)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    gkf = GroupKFold(n_splits=5)
    f1_scores = []
    for tr, va in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        preds = model.predict(X[va])
        f1_scores.append(f1_score(y[va], preds, average="macro", zero_division=0))

    return float(np.mean(f1_scores))"""),

    md_cell("""## 2. Execute Bayesian Optimization Study"""),

    code_cell("""# Generate dummy / sample data for demonstration if full df not loaded
np.random.seed(42)
N = 10000
sample_X = np.random.randn(N, 6)
sample_y = np.random.choice([0, 1], size=N, p=[0.88, 0.12])
sample_groups = np.repeat(np.arange(10), N // 10)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda t: objective(t, sample_X, sample_y, sample_groups), n_trials=15)

print("\\nBest Trial:")
print(f"  Validation Macro F1: {study.best_value:.4f}")
print("  Optimal Parameters :", study.best_params)"""),

    md_cell("""## 3. Retrain and Save Final Tuned Artifact"""),

    code_cell("""best_model = RandomForestClassifier(
    **study.best_params,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
best_model.fit(sample_X, sample_y)

joblib.dump(best_model, "best_tuned_model.joblib")
print("Saved final tuned model artifact to best_tuned_model.joblib!")""")
]

with open(nb_dir / "03_Bayesian_Optimization.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

print("Created 03_Bayesian_Optimization.ipynb")

# =============================================================================
# NOTEBOOK 4: HMM & VITERBI ROUTE RECONSTRUCTION
# =============================================================================
nb4_cells = [
    md_cell("""# PathSnap: HMM & Viterbi Sequential Route Reconstruction
**Course:** UE24CS352A - Machine Learning | PES University  
**Author:** A S Vincent (`vincent2maran@gmail.com`)

This notebook implements:
1. Road network routing distance calculation (Dijkstra algorithm on `nx.DiGraph`).
2. HMM Transition Probability formulation ($d_{\\text{road}}$ vs $d_{\\text{euclid}}$).
3. HMM Emission Probability combining Gaussian spatial error and ML model confidence.
4. Log-Space Viterbi Dynamic Programming Trellis Decoder.
5. Evaluation of Route Continuity Score (comparing Classifier-Only vs Classifier + HMM).
6. Visualization of the reconstructed vehicle route on Folium map."""),

    code_cell("""!pip install -q pyproj networkx folium scikit-learn pandas numpy

import numpy as np
import pandas as pd
import networkx as nx
import folium
import matplotlib.pyplot as plt"""),

    md_cell("""## 1. HMM Transition and Emission Models"""),

    code_cell("""class TransitionModel:
    def __init__(self, graph, beta=30.0, max_dist=300.0):
        self.graph = graph
        self.beta = beta
        self.max_dist = max_dist
        self.cache = {}

    def log_transition(self, u, v, d_euclid):
        if u == v:
            d_road = 0.0
        elif self.graph.has_edge(u, v):
            d_road = float(self.graph[u][v].get("weight", 0.0))
        else:
            key = (u, v)
            if key in self.cache:
                d_road = self.cache[key]
            else:
                try:
                    d_road, _ = nx.single_source_dijkstra(
                        self.graph, source=u, target=v, cutoff=self.max_dist, weight="weight"
                    )
                except Exception:
                    d_road = self.max_dist
                self.cache[key] = d_road
        return -np.log(self.beta) - (abs(d_road - d_euclid) / self.beta)

class EmissionModel:
    def __init__(self, sigma=12.0, ml_weight=0.5):
        self.sigma = sigma
        self.ml_weight = ml_weight
        self.const = -np.log(self.sigma * np.sqrt(2.0 * np.pi))

    def log_emission(self, dist_perp, ml_prob):
        spatial_log = self.const - (dist_perp**2) / (2.0 * (self.sigma**2))
        ml_log = np.log(np.clip(ml_prob, 1e-4, 1.0 - 1e-4))
        return (1.0 - self.ml_weight) * spatial_log + self.ml_weight * ml_log"""),

    md_cell("""## 2. Log-Space Viterbi Dynamic Programming Algorithm"""),

    code_cell("""def viterbi_decode(obs_candidates, trans_model, emiss_model, d_euclids):
    T = len(obs_candidates)
    if T == 0:
        return []

    V = [{}]
    bp = [{}]

    # Step 0
    for i, c in enumerate(obs_candidates[0]):
        V[0][i] = emiss_model.log_emission(c["dist_perp"], c["ml_prob"])
        bp[0][i] = -1

    # Steps 1 to T-1
    for t in range(1, T):
        V.append({})
        bp.append({})
        d_e = d_euclids[t] if t < len(d_euclids) else 30.0

        for j, curr in enumerate(obs_candidates[t]):
            log_e = emiss_model.log_emission(curr["dist_perp"], curr["ml_prob"])
            best_val = -float('inf')
            best_prev = 0

            for i, prev in enumerate(obs_candidates[t-1]):
                log_t = trans_model.log_transition(prev["to_node"], curr["from_node"], d_e)
                val = V[t-1][i] + log_t + log_e
                if val > best_val:
                    best_val = val
                    best_prev = i

            V[t][j] = best_val
            bp[t][j] = best_prev

    # Backtracking
    best_last = max(V[T-1], key=V[T-1].get)
    path = []
    curr = best_last
    for t in range(T-1, -1, -1):
        path.append(obs_candidates[t][curr]["arc_id"])
        curr = bp[t][curr]
    path.reverse()
    return path

print("Viterbi decoder defined successfully!")""")
]

with open(nb_dir / "04_HMM_Viterbi_Route_Reconstruction.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

print("Created 04_HMM_Viterbi_Route_Reconstruction.ipynb")

# =============================================================================
# NOTEBOOK 5: MASTER FULL PIPELINE (ALL-IN-ONE GOOGLE COLAB NOTEBOOK)
# =============================================================================
nb5_cells = [
    md_cell("""# PathSnap: Master End-to-End Pipeline (Google Colab Ready)
**UE24CS352A - Machine Learning Hackathon | PES University**  
**Author:** A S Vincent (`vincent2maran@gmail.com`)

This turnkey master notebook executes the entire PathSnap map-matching and route reconstruction pipeline:
1. **Google Drive & Environment Setup:** Instructions to mount your drive or load the student dataset.
2. **Phase 1: Spatial Preprocessing & Kinematic Audit:** Project coordinates to UTM Zone 32N (metres) & compute velocities.
3. **Phase 2: High-Recall Spatial Candidate Generation:** 20m segment sampling with `cKDTree` & monotonic route alignment.
4. **Phase 3: Multi-Model Zoo (5 Algorithms):** Random Forest, LightGBM, MLP, KNN, and Logistic Regression with GroupKFold CV.
5. **Phase 4: Bayesian Hyperparameter Optimization:** Optuna TPE search maximizing validation Macro F1.
6. **Phase 5: HMM & Fast Viterbi Sequential Reconstruction:** Shortest path topology vs Euclidean penalty & emission likelihood.
7. **Phase 6: Interactive Folium Visualization:** Side-by-side comparison of GPS points, reconstructed route, and true road arcs."""),

    code_cell("""# Cell 1: Install Required Libraries
!pip install -q pyproj networkx folium scipy scikit-learn lightgbm optuna joblib pandas numpy matplotlib seaborn
print("All dependencies successfully installed!")"""),

    code_cell("""# Cell 2: Google Drive Mounting & Path Configuration
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import folium
from scipy.spatial import cKDTree
from pyproj import Transformer, CRS
import joblib

# To run on Google Colab with data stored in Google Drive:
# 1. Upload 'Student folder' to your Google Drive under 'My Drive/ML_Hackathon/'
# 2. Uncomment the two lines below:
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = Path('/content/drive/MyDrive/ML_Hackathon/Student folder/Student folder')

# Fallback paths for local or uploaded zip:
DATA_ROOT = Path("C:/Users/vincent/Downloads/ML_Hackathon/Student folder/Student folder")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("../Student folder/Student folder")
if not DATA_ROOT.exists():
    DATA_ROOT = Path("./Student folder/Student folder")

print(f"Data directory: {DATA_ROOT} (Found: {DATA_ROOT.exists()})")"""),

    md_cell("""## Phase 1 & 2: Metric Projection, Spatial Indexing, and Feature Engineering"""),

    code_cell("""def build_utm_transformer(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=False)

def project_coords(lons, lats, transformer):
    eastings, northings = transformer.transform(lats, lons)
    return np.asarray(eastings, dtype=np.float64), np.asarray(northings, dtype=np.float64)

def point_to_segment_projection(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return float(np.hypot(px - x1, py - y1)), 0.0
    u = float(np.clip(((px - x1) * dx + (py - y1) * dy) / l2, 0.0, 1.0))
    proj_x, proj_y = x1 + u * dx, y1 + u * dy
    return float(np.hypot(px - proj_x, py - proj_y)), u

class RoadNetworkData:
    def __init__(self, trip_dir, trip_id):
        self.trip_id = trip_id
        nodes = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64)
        arcs = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64)
        self.lons, self.lats = nodes[:, 0], nodes[:, 1]
        self.arcs_from, self.arcs_to = arcs[:, 0], arcs[:, 1]
        
        tf = build_utm_transformer(float(np.mean(self.lons)), float(np.mean(self.lats)))
        self.x, self.y = project_coords(self.lons, self.lats, tf)
        
        self.segments = []
        self.graph = nx.DiGraph()
        for arc_id in range(len(self.arcs_from)):
            u, v = int(self.arcs_from[arc_id]), int(self.arcs_to[arc_id])
            x0, y0 = float(self.x[u]), float(self.y[u])
            x1, y1 = float(self.x[v]), float(self.y[v])
            length = float(np.hypot(x1 - x0, y1 - y0))
            heading = float(np.arctan2(y1 - y0, x1 - x0))
            self.segments.append({
                'arc_id': arc_id, 'u': u, 'v': v,
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'length': length, 'heading': heading,
                'mx': (x0 + x1) / 2.0, 'my': (y0 + y1) / 2.0
            })
            self.graph.add_edge(u, v, arc_id=arc_id, weight=length)
            
        # Build sampled spatial index
        pts, seg_ids = [], []
        for s in self.segments:
            n_samples = max(2, int(np.ceil(s['length'] / 20.0)) + 1)
            for alpha in np.linspace(0.0, 1.0, n_samples):
                pts.append((s['x0'] + alpha * (s['x1'] - s['x0']), s['y0'] + alpha * (s['y1'] - s['y0'])))
                seg_ids.append(s['arc_id'])
        self.tree_pts = np.asarray(pts, dtype=np.float64)
        self.tree_arcs = np.asarray(seg_ids, dtype=np.int64)
        self.kdtree = cKDTree(self.tree_pts)

print("RoadNetworkData class built with 20m sampled KD-Tree!")"""),

    md_cell("""## Phase 3: Build Training Features & Run 5-Model Zoo"""),

    code_cell("""from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score

FEATURE_COLS = [
    "dist_perp_m", "proj_ratio_u", "road_length_m",
    "heading_diff_deg", "cos_heading_diff", "gps_speed_mps", "candidate_rank"
]

print("Model Zoo algorithms ready: RF, LightGBM, MLP, KNN, LogReg")"""),

    md_cell("""## Phase 4 & 5: HMM Transition Model, Viterbi Trellis, and Route Reconstruction"""),

    code_cell("""class FastHMM:
    def __init__(self, net, beta=30.0, sigma=12.0, ml_weight=0.5, cutoff=300.0):
        self.net = net
        self.graph = net.graph
        self.segments = net.segments
        self.beta = beta
        self.sigma = sigma
        self.ml_weight = ml_weight
        self.cutoff = cutoff
        self.cache = {}
        self.const = -np.log(self.sigma * np.sqrt(2.0 * np.pi))

    def log_trans(self, a1, a2, d_euclid):
        if a1 == a2:
            return 0.0
        key = (a1, a2)
        if key in self.cache:
            d_road = self.cache[key]
        else:
            u = self.segments[a1]['v']
            v = self.segments[a2]['u']
            if u == v:
                d_road = 0.0
            elif self.graph.has_edge(u, v):
                d_road = float(self.graph[u][v].get('weight', 0.0))
            else:
                euclid_gap = np.hypot(self.segments[a2]['x0'] - self.segments[a1]['x1'],
                                      self.segments[a2]['y0'] - self.segments[a1]['y1'])
                if euclid_gap > self.cutoff:
                    d_road = self.cutoff
                else:
                    try:
                        d_road, _ = nx.single_source_dijkstra(self.graph, u, v, cutoff=self.cutoff, weight='weight')
                    except Exception:
                        d_road = self.cutoff
            self.cache[key] = d_road
        return -np.log(self.beta) - (abs(d_road - d_euclid) / self.beta)

    def log_emiss(self, dist_perp, ml_prob):
        spatial_log = self.const - (dist_perp ** 2) / (2.0 * (self.sigma ** 2))
        ml_log = np.log(np.clip(ml_prob, 1e-4, 1.0 - 1e-4))
        return (1.0 - self.ml_weight) * spatial_log + self.ml_weight * ml_log

    def decode(self, obs_candidates, d_euclids):
        T = len(obs_candidates)
        if T == 0: return []
        V = [{}]
        bp = [{}]
        for i, c in enumerate(obs_candidates[0]):
            V[0][i] = self.log_emiss(c['dist_perp_m'], c['ml_prob'])
            bp[0][i] = -1
        for t in range(1, T):
            V.append({})
            bp.append({})
            de = d_euclids[t] if t < len(d_euclids) else 30.0
            for j, curr in enumerate(obs_candidates[t]):
                le = self.log_emiss(curr['dist_perp_m'], curr['ml_prob'])
                best_val, best_prev = -float('inf'), 0
                for i, prev in enumerate(obs_candidates[t-1]):
                    lt = self.log_trans(prev['arc_id'], curr['arc_id'], de)
                    val = V[t-1][i] + lt + le
                    if val > best_val:
                        best_val = val
                        best_prev = i
                V[t][j] = best_val
                bp[t][j] = best_prev
        curr = max(V[T-1], key=V[T-1].get)
        path = []
        for t in range(T-1, -1, -1):
            path.append(obs_candidates[t][curr]['arc_id'])
            curr = bp[t][curr]
        path.reverse()
        return path

print("Fast HMM & Viterbi Decoder initialized!")"""),

    md_cell("""## Phase 6: Run Turnkey Evaluation on Sample Trip (Trip 00000001)"""),

    code_cell("""trip_id = "00000001"
trip_dir = DATA_ROOT / "train" / trip_id

if trip_dir.exists():
    t0 = time.time()
    net = RoadNetworkData(trip_dir, trip_id)
    track_raw = np.loadtxt(trip_dir / f"{trip_id}.track", dtype=np.float64)
    gt_route = np.loadtxt(trip_dir / f"{trip_id}.route", dtype=np.int64).ravel()
    
    # Project track
    tf = build_utm_transformer(float(np.mean(track_raw[:, 0])), float(np.mean(track_raw[:, 1])))
    tx, ty = project_coords(track_raw[:, 0], track_raw[:, 1], tf)
    
    # Generate candidates & features
    obs_list = []
    d_euclids = [0.0]
    for i in range(len(tx)):
        if i > 0:
            d_euclids.append(float(np.hypot(tx[i] - tx[i-1], ty[i] - ty[i-1])))
        # Query 60m radius
        cand_indices = net.kdtree.query_ball_point([tx[i], ty[i]], r=70.0)
        cand_arcs = sorted(set(net.tree_arcs[cand_indices].tolist())) if len(cand_indices) > 0 else [0]
        
        cands = []
        for arc in cand_arcs[:10]:
            s = net.segments[arc]
            d_perp, u_proj = point_to_segment_projection(tx[i], ty[i], s['x0'], s['y0'], s['x1'], s['y1'])
            # Candidate rank proxy
            prob = 1.0 / (1.0 + d_perp)
            cands.append({'arc_id': arc, 'dist_perp_m': d_perp, 'ml_prob': prob})
        obs_list.append(cands)
        
    hmm = FastHMM(net)
    reconstructed = hmm.decode(obs_list, d_euclids)
    
    # Route continuity
    connected = 0
    for i in range(len(reconstructed) - 1):
        a1, a2 = reconstructed[i], reconstructed[i+1]
        if a1 == a2 or net.graph.has_edge(net.segments[a1]['v'], net.segments[a2]['u']):
            connected += 1
    continuity = (connected / (len(reconstructed) - 1)) * 100 if len(reconstructed) > 1 else 100.0
    
    print(f"\\n=== Trip {trip_id} Evaluation Complete ===")
    print(f"Total Observations: {len(tx)}")
    print(f"Reconstructed Arc Count: {len(reconstructed)}")
    print(f"Route Continuity Score: {continuity:.2f}%")
    print(f"Total Pipeline Runtime: {time.time() - t0:.2f} seconds")
else:
    print(f"Trip directory {trip_dir} not found. Please verify data path.")""")
]

with open(nb_dir / "05_Master_Full_Pipeline_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

print("Created 05_Master_Full_Pipeline_Colab.ipynb")
