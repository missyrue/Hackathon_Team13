"""
GPS Trajectory Loader and Diagnostic Auditor
---------------------------------------------
Loads .track files, computes kinematic features (distance, delta t, speed),
and audits sampling frequency and GPS anomalies.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def haversine_distance_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance in metres between successive points."""
    r = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * r * np.arcsin(np.sqrt(a))


def load_track(filepath: str | Path, trip_id: str | None = None) -> pd.DataFrame:
    """
    Loads a whitespace-delimited .track file containing:
    Column 0: longitude (deg)
    Column 1: latitude (deg)
    Column 2: timestamp (seconds)
    """
    filepath = Path(filepath)
    if trip_id is None:
        trip_id = filepath.stem

    data = np.loadtxt(filepath, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    lons = data[:, 0]
    lats = data[:, 1]
    timestamps = data[:, 2]
    n_pts = len(lons)

    # Compute step differences
    dt_s = np.zeros(n_pts, dtype=np.float64)
    dist_m = np.zeros(n_pts, dtype=np.float64)
    speed_mps = np.zeros(n_pts, dtype=np.float64)

    if n_pts > 1:
        dt_s[1:] = np.diff(timestamps)
        dist_m[1:] = haversine_distance_m(lons[:-1], lats[:-1], lons[1:], lats[1:])
        valid_dt = dt_s[1:] > 0
        speed_mps[1:][valid_dt] = dist_m[1:][valid_dt] / dt_s[1:][valid_dt]

    # Diagnostics
    is_regular = np.abs(dt_s - 1.0) <= 0.25
    is_regular[0] = True
    is_gap = dt_s > 2.5
    is_jump = speed_mps > 50.0  # > 180 km/h

    return pd.DataFrame({
        "trip_id": trip_id,
        "observation_idx": np.arange(n_pts, dtype=int),
        "lon": lons,
        "lat": lats,
        "timestamp_s": timestamps,
        "dt_s": dt_s,
        "distance_m": dist_m,
        "speed_mps": speed_mps,
        "is_regular": is_regular,
        "is_gap": is_gap,
        "is_jump": is_jump,
    })
