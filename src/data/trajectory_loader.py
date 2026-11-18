"""
Trajectory loading and preprocessing for PathSnap ``.track`` files.

The dataset stores one observation per row as longitude, latitude, and elapsed
seconds.  This module keeps input row order, derives motion/time diagnostics,
and summarizes directories of trips for the training-data audit.

Raw coordinates are WGS84 degrees. Distances and speeds below use a haversine
distance in metres, which is suitable for the short trip-local anomaly audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0
TRACK_COLUMNS = ("lon", "lat", "timestamp_s")


def _haversine_distances_m(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return distances between each observation and its predecessor, in metres."""
    if len(lon) < 2:
        return np.empty(0, dtype=np.float64)

    lon1 = np.radians(lon[:-1])
    lon2 = np.radians(lon[1:])
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # Guard against floating-point roundoff beyond [0, 1].
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_track(
    filepath: str | Path,
    trip_id: str | None = None,
    *,
    sampling_interval_s: float = 1.0,
    regular_tolerance_s: float = 0.25,
) -> pd.DataFrame:
    """Load one whitespace-delimited ``.track`` file and add motion diagnostics.

    Parameters
    ----------
    filepath:
        Path to a ``.track`` file. Each non-empty row must contain exactly
        longitude, latitude, and elapsed time in seconds.
    trip_id:
        Optional trip identifier. Defaults to the containing directory name.
    sampling_interval_s:
        Nominal sampling interval used to mark regular time steps.
    regular_tolerance_s:
        Allowed absolute deviation from the nominal interval, in seconds.

    Returns
    -------
    pandas.DataFrame
        One row per original observation, with columns ``trip_id``,
        ``observation_idx``, ``lon``, ``lat``, ``timestamp_s``, ``dt_s``,
        ``distance_m``, ``speed_mps``, ``is_time_step_regular``, and
        ``is_stationary``. The first row has no preceding-step diagnostics.

    Raises
    ------
    ValueError
        If a row has the wrong number of fields, non-numeric/non-finite values,
        out-of-range coordinates, or timestamps that do not strictly increase.
    """
    path = Path(filepath)
    if sampling_interval_s <= 0:
        raise ValueError("sampling_interval_s must be greater than zero")
    if regular_tolerance_s < 0:
        raise ValueError("regular_tolerance_s cannot be negative")

    values: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected 3 columns "
                    f"(longitude latitude timestamp), found {len(fields)}"
                )
            try:
                lon, lat, timestamp_s = map(float, fields)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: expected numeric values") from exc
            if not np.isfinite([lon, lat, timestamp_s]).all():
                raise ValueError(f"{path}:{line_number}: values must be finite")
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"{path}:{line_number}: longitude outside [-180, 180]")
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"{path}:{line_number}: latitude outside [-90, 90]")
            values.append((lon, lat, timestamp_s))

    if not values:
        raise ValueError(f"No observations found in {path}")

    data = np.asarray(values, dtype=np.float64)
    lon, lat, timestamp_s = data.T
    if len(timestamp_s) > 1 and np.any(np.diff(timestamp_s) <= 0):
        bad_row = int(np.flatnonzero(np.diff(timestamp_s) <= 0)[0]) + 2
        raise ValueError(
            f"{path}:{bad_row}: timestamps must strictly increase; "
            "input row order is preserved"
        )

    n = len(data)
    dt_s = np.full(n, np.nan, dtype=np.float64)
    distance_m = np.full(n, np.nan, dtype=np.float64)
    speed_mps = np.full(n, np.nan, dtype=np.float64)
    is_regular = np.zeros(n, dtype=bool)
    is_stationary = np.zeros(n, dtype=bool)

    if n > 1:
        dt = np.diff(timestamp_s)
        distance = _haversine_distances_m(lon, lat)
        speed = distance / dt
        dt_s[1:] = dt
        distance_m[1:] = distance
        speed_mps[1:] = speed
        is_regular[1:] = np.abs(dt - sampling_interval_s) <= regular_tolerance_s
        # Keep this feature descriptive. A multi-sample dwell cluster is
        # summarized separately so one slow interval is not called a cluster.
        is_stationary[1:] = speed <= 0.5

    return pd.DataFrame(
        {
            "trip_id": str(trip_id if trip_id is not None else path.parent.name),
            "observation_idx": np.arange(n, dtype=np.int64),
            "lon": lon,
            "lat": lat,
            "timestamp_s": timestamp_s,
            "dt_s": dt_s,
            "distance_m": distance_m,
            "speed_mps": speed_mps,
            "is_time_step_regular": is_regular,
            "is_stationary": is_stationary,
        }
    )


def summarize_track(
    track: pd.DataFrame,
    *,
    jump_speed_mps: float = 50.0,
    stationary_speed_mps: float = 0.5,
    stationary_min_observations: int = 3,
) -> dict[str, Any]:
    """Summarize trip length, duration, sampling regularity, and anomalies.

    GPS jumps are intervals whose observed speed exceeds ``jump_speed_mps``.
    A stationary cluster is a run of at least ``stationary_min_observations``
    consecutive low-speed intervals. The first observation has no interval.
    """
    if jump_speed_mps <= 0:
        raise ValueError("jump_speed_mps must be greater than zero")
    if stationary_speed_mps < 0:
        raise ValueError("stationary_speed_mps cannot be negative")
    if stationary_min_observations < 1:
        raise ValueError("stationary_min_observations must be at least one")
    if track.empty:
        raise ValueError("Cannot summarize an empty trajectory")

    valid_intervals = track["dt_s"].notna()
    dt = track.loc[valid_intervals, "dt_s"].to_numpy(dtype=np.float64)
    speed = track.loc[valid_intervals, "speed_mps"].to_numpy(dtype=np.float64)
    regular = track.loc[valid_intervals, "is_time_step_regular"].to_numpy(dtype=bool)
    stationary = speed <= stationary_speed_mps

    # Count consecutive low-speed intervals, not individual GPS observations.
    stationary_cluster_count = 0
    run_length = 0
    for is_slow in stationary:
        if is_slow:
            run_length += 1
        else:
            if run_length >= stationary_min_observations:
                stationary_cluster_count += 1
            run_length = 0
    if run_length >= stationary_min_observations:
        stationary_cluster_count += 1

    distance_values = track.loc[valid_intervals, "distance_m"].to_numpy(dtype=np.float64)
    n_observations = int(len(track))
    return {
        "trip_id": str(track["trip_id"].iloc[0]),
        "observation_count": n_observations,
        "duration_s": float(track["timestamp_s"].iloc[-1] - track["timestamp_s"].iloc[0]),
        "median_dt_s": float(np.median(dt)) if len(dt) else np.nan,
        "regular_interval_fraction": float(np.mean(regular)) if len(regular) else np.nan,
        "irregular_interval_count": int(np.count_nonzero(~regular)),
        "long_gap_count": int(np.count_nonzero(dt > 1.5)),
        "max_gap_s": float(np.max(dt)) if len(dt) else np.nan,
        "path_distance_km": float(np.sum(distance_values) / 1000.0),
        "median_speed_mps": float(np.median(speed)) if len(speed) else np.nan,
        "max_speed_mps": float(np.max(speed)) if len(speed) else np.nan,
        "gps_jump_count": int(np.count_nonzero(speed > jump_speed_mps)),
        "stationary_interval_count": int(np.count_nonzero(stationary)),
        "stationary_cluster_count": int(stationary_cluster_count),
    }


def audit_track_directory(
    train_dir: str | Path,
    *,
    jump_speed_mps: float = 50.0,
    stationary_speed_mps: float = 0.5,
    stationary_min_observations: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all ``.track`` files recursively under a training-data directory.

    Returns ``(per_trip, dataset_summary)``. This deliberately accepts a
    training directory path rather than discovering or reading test labels.
    """
    root = Path(train_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Training track directory does not exist: {root}")
    paths = sorted(root.rglob("*.track"))
    if not paths:
        raise FileNotFoundError(f"No .track files found under training directory: {root}")

    summaries = []
    for path in paths:
        track = load_track(path)
        summaries.append(
            summarize_track(
                track,
                jump_speed_mps=jump_speed_mps,
                stationary_speed_mps=stationary_speed_mps,
                stationary_min_observations=stationary_min_observations,
            )
        )
    per_trip = pd.DataFrame(summaries)
    dataset_summary = pd.DataFrame(
        [
            {
                "trip_count": int(len(per_trip)),
                "total_observation_count": int(per_trip["observation_count"].sum()),
                "time_interval_count": int(per_trip["observation_count"].sum() - len(per_trip)),
                "median_observations_per_trip": float(per_trip["observation_count"].median()),
                "median_duration_s": float(per_trip["duration_s"].median()),
                "median_regular_interval_fraction": float(
                    per_trip["regular_interval_fraction"].median()
                ),
                "overall_regular_interval_fraction": float(
                    1.0
                    - per_trip["irregular_interval_count"].sum()
                    / max(int(per_trip["observation_count"].sum() - len(per_trip)), 1)
                ),
                "irregular_interval_count": int(per_trip["irregular_interval_count"].sum()),
                "long_gap_count": int(per_trip["long_gap_count"].sum()),
                "gps_jump_count": int(per_trip["gps_jump_count"].sum()),
                "stationary_cluster_count": int(per_trip["stationary_cluster_count"].sum()),
            }
        ]
    )
    return per_trip, dataset_summary


def _main() -> None:
    """Run the training-track audit and write reproducible CSV summaries."""
    import argparse

    parser = argparse.ArgumentParser(description="Load and audit training .track trajectories.")
    parser.add_argument("train_dir", help="Directory containing training trip folders")
    parser.add_argument(
        "--output-dir",
        default="data/processed/trajectory_audit",
        help="Directory for per-trip inventory and aggregate summary CSVs",
    )
    parser.add_argument(
        "--jump-speed-mps",
        type=float,
        default=50.0,
        help="Observed speed above which an interval is flagged as a GPS jump",
    )
    args = parser.parse_args()

    per_trip, dataset_summary = audit_track_directory(
        args.train_dir, jump_speed_mps=args.jump_speed_mps
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_trip.to_csv(output_dir / "trajectory_inventory.csv", index=False)
    dataset_summary.to_csv(output_dir / "trajectory_dataset_summary.csv", index=False)
    print(dataset_summary.to_string(index=False))
    print(f"Per-trip inventory: {output_dir / 'trajectory_inventory.csv'}")
    print(f"Dataset summary: {output_dir / 'trajectory_dataset_summary.csv'}")


if __name__ == "__main__":
    _main()

