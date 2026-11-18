"""
Candidate Road Segment Generation & Feature Extraction
-------------------------------------------------------
Implements high-recall spatial search using cKDTree over road segments,
monotonic chronological ground-truth route alignment, and feature engineering.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.spatial_projection import build_transformer, project_coords
from src.data.road_network import RoadNetwork


def point_to_segment_projection(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float
) -> tuple[float, float]:
    """Computes exact perpendicular distance and along-track ratio u in [0, 1]."""
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return float(np.hypot(px - x1, py - y1)), 0.0

    u = ((px - x1) * dx + (py - y1) * dy) / l2
    u_clamped = max(0.0, min(1.0, u))
    proj_x = x1 + u_clamped * dx
    proj_y = y1 + u_clamped * dy
    dist = float(np.hypot(px - proj_x, py - proj_y))
    return dist, float(u_clamped)


class RoadSegmentSpatialIndex:
    """Samples points along road segments to ensure high recall for long roads."""

    def __init__(self, network: RoadNetwork, sample_step_m: float = 20.0):
        self.network = network
        self.sample_step_m = float(sample_step_m)
        self.segment_by_id = {s.arc_id: s for s in network.segments}

        pts = []
        seg_ids = []

        for seg in network.segments:
            seg_len = seg.length_m
            n_samples = max(2, int(np.ceil(seg_len / self.sample_step_m)) + 1)
            alphas = np.linspace(0.0, 1.0, n_samples)
            for a in alphas:
                x = seg.x_from + a * (seg.x_to - seg.x_from)
                y = seg.y_from + a * (seg.y_to - seg.y_from)
                pts.append((x, y))
                seg_ids.append(seg.arc_id)

        self.kdtree = cKDTree(np.array(pts, dtype=np.float64))
        self.seg_ids = np.array(seg_ids, dtype=np.int64)

    def query_candidate_arcs(self, px: float, py: float, radius_m: float = 60.0) -> list[int]:
        search_radius = radius_m + self.sample_step_m / 2.0
        indices = self.kdtree.query_ball_point([px, py], r=search_radius)
        if not indices:
            _, nearest_idx = self.kdtree.query([px, py], k=1)
            indices = [nearest_idx]
        return sorted(set(self.seg_ids[indices].tolist()))


def align_track_to_ordered_route(
    gps_x: np.ndarray,
    gps_y: np.ndarray,
    route_arcs: np.ndarray,
    segment_by_id: dict,
    timestamps_s: np.ndarray | None = None,
) -> np.ndarray:
    """
    Globally aligns GPS observations to the chronological reference route
    using dynamic programming trellis alignment, strictly preventing false
    forward jumps across parallel streets, roundabouts, or interchanges.
    """
    route = np.asarray(route_arcs, dtype=np.int64).reshape(-1)
    T = len(gps_x)
    K = len(route)
    if T == 0 or K == 0:
        return np.empty(0, dtype=np.int64)

    route_segs = [segment_by_id[int(r)] for r in route]
    dp = np.full((T, K), np.inf, dtype=np.float32)
    backptr = np.zeros((T, K), dtype=np.int32)

    # Initialize at t = 0: vehicle begins near start of route
    init_window = min(6, K)
    for k in range(init_window):
        d, _ = point_to_segment_projection(
            gps_x[0], gps_y[0],
            route_segs[k].x_from, route_segs[k].y_from,
            route_segs[k].x_to, route_segs[k].y_to
        )
        dp[0, k] = d

    # Dynamic programming forward pass
    for t in range(1, T):
        px, py = float(gps_x[t]), float(gps_y[t])
        dt = 1.0
        if timestamps_s is not None and t < len(timestamps_s):
            dt = max(1.0, float(timestamps_s[t] - timestamps_s[t - 1]))
        max_step = min(K, max(4, int(dt * 3.0)))

        valid_prev = np.where(dp[t - 1] < np.inf)[0]
        if len(valid_prev) == 0:
            valid_prev = np.array([0], dtype=int)
            dp[t - 1, 0] = 0.0

        min_k = max(0, int(np.min(valid_prev)))
        max_k = min(K, int(np.max(valid_prev)) + max_step + 1)

        for k in range(min_k, max_k):
            d, _ = point_to_segment_projection(
                px, py,
                route_segs[k].x_from, route_segs[k].y_from,
                route_segs[k].x_to, route_segs[k].y_to
            )
            best_c = np.inf
            best_p = k

            start_p = max(0, k - max_step)
            for p in range(start_p, k + 1):
                if dp[t - 1, p] < np.inf:
                    step_penalty = 2.0 * (k - p) if k != p else 0.0
                    cost = dp[t - 1, p] + d + step_penalty
                    if cost < best_c:
                        best_c = cost
                        best_p = p

            if best_c < np.inf:
                dp[t, k] = best_c
                backptr[t, k] = best_p

    # Backtracking optimal trajectory
    aligned = np.zeros(T, dtype=np.int64)
    best_last_k = int(np.argmin(dp[T - 1, :]))
    curr_k = best_last_k
    for t in range(T - 1, -1, -1):
        aligned[t] = route[curr_k]
        curr_k = backptr[t, curr_k]

    return aligned


def generate_features(
    track: pd.DataFrame,
    network: RoadNetwork,
    route: np.ndarray | None = None,
    candidate_radius_m: float = 80.0,
    top_k: int = 15,
    trip_id: str | None = None,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Generates observation-candidate feature rows with temporal and relative ranking features."""
    if trip_id is None:
        trip_id = str(getattr(network, "trip_id", track.get("trip_id", [""])[0]))

    tf = build_transformer(float(np.mean(network.nodes_lon)), float(np.mean(network.nodes_lat)))
    gps_x, gps_y = project_coords(track["lon"].values, track["lat"].values, tf)
    n_obs = len(track)

    # Heading calculation
    dx = np.diff(gps_x, append=gps_x[-1])
    dy = np.diff(gps_y, append=gps_y[-1])
    gps_headings = np.arctan2(dy, dx)

    speeds = track["speed_mps"].values if "speed_mps" in track else np.zeros(n_obs)
    timestamps = track["timestamp_s"].values if "timestamp_s" in track else np.arange(n_obs, dtype=float)
    spatial_index = RoadSegmentSpatialIndex(network)

    true_arcs = None
    if route is not None:
        true_arcs = align_track_to_ordered_route(
            gps_x, gps_y, route, spatial_index.segment_by_id, timestamps_s=timestamps
        )

    rows = []
    miss_count = 0

    for i in range(n_obs):
        px, py = float(gps_x[i]), float(gps_y[i])
        px_prev = float(gps_x[max(0, i - 1)])
        py_prev = float(gps_y[max(0, i - 1)])
        px_next = float(gps_x[min(n_obs - 1, i + 1)])
        py_next = float(gps_y[min(n_obs - 1, i + 1)])
        heading = float(gps_headings[i])
        true_arc = int(true_arcs[i]) if true_arcs is not None else None

        cand_arcs = spatial_index.query_candidate_arcs(px, py, radius_m=candidate_radius_m)

        scored = []
        for arc_id in cand_arcs:
            seg = spatial_index.segment_by_id[arc_id]
            dist_perp, ratio = point_to_segment_projection(px, py, seg.x_from, seg.y_from, seg.x_to, seg.y_to)
            dist_prev, ratio_prev = point_to_segment_projection(px_prev, py_prev, seg.x_from, seg.y_from, seg.x_to, seg.y_to)
            dist_next, ratio_next = point_to_segment_projection(px_next, py_next, seg.x_from, seg.y_from, seg.x_to, seg.y_to)
            h_diff = float(abs((heading - seg.heading_rad + np.pi) % (2.0 * np.pi) - np.pi))
            if dist_perp <= candidate_radius_m:
                scored.append((arc_id, dist_perp, ratio, seg, h_diff, dist_prev, ratio_prev, dist_next, ratio_next))

        if not scored:  # fallback to nearest if none in radius
            seg = spatial_index.segment_by_id[cand_arcs[0]]
            dist_perp, ratio = point_to_segment_projection(px, py, seg.x_from, seg.y_from, seg.x_to, seg.y_to)
            dist_prev, ratio_prev = point_to_segment_projection(px_prev, py_prev, seg.x_from, seg.y_from, seg.x_to, seg.y_to)
            dist_next, ratio_next = point_to_segment_projection(px_next, py_next, seg.x_from, seg.y_from, seg.x_to, seg.y_to)
            h_diff = float(abs((heading - seg.heading_rad + np.pi) % (2.0 * np.pi) - np.pi))
            scored.append((cand_arcs[0], dist_perp, ratio, seg, h_diff, dist_prev, ratio_prev, dist_next, ratio_next))

        scored.sort(key=lambda item: (item[1], item[0]))
        selected = scored[:top_k]

        selected_arc_ids = [s[0] for s in selected]
        if true_arc is not None and true_arc not in selected_arc_ids:
            miss_count += 1

        min_d = selected[0][1]
        min_h = min(s[4] for s in selected)
        cand_count = len(selected)

        d_arr = np.array([s[1] for s in selected])
        exp_d = np.exp(-(d_arr - min_d) / 15.0)
        prob_d = exp_d / (np.sum(exp_d) + 1e-7)

        for rank, (arc_id, dist_perp, ratio, seg, h_diff, d_p, u_p, d_n, u_n) in enumerate(selected):
            row = {
                "trip_id": trip_id,
                "observation_idx": i,
                "arc_id": int(arc_id),
                "dist_perp_m": float(dist_perp),
                "proj_ratio_u": float(ratio),
                "road_length_m": float(seg.length_m),
                "dist_prev_m": float(d_p),
                "dist_next_m": float(d_n),
                "u_diff_prev": float(ratio - u_p),
                "u_diff_next": float(u_n - ratio),
                "heading_diff_deg": float(np.degrees(h_diff)),
                "cos_heading_diff": float(np.cos(h_diff)),
                "abs_cos_heading": float(abs(np.cos(h_diff))),
                "gps_speed_mps": float(speeds[i]),
                "candidate_rank": int(rank),
                "dist_minus_min": float(dist_perp - min_d),
                "dist_ratio": float(dist_perp / (min_d + 1.0)),
                "heading_minus_min": float(h_diff - min_h),
                "spatial_softmax": float(prob_d[rank]),
                "speed_x_cos_heading": float(speeds[i] * np.cos(h_diff)),
                "candidate_count": int(cand_count),
            }
            if true_arc is not None:
                row["is_true_match"] = int(arc_id == true_arc)
            rows.append(row)

    df_feats = pd.DataFrame(rows)
    if return_diagnostics:
        diag = {
            "observation_count": n_obs,
            "top_k_miss_count": miss_count,
            "candidate_miss_rate": (miss_count / n_obs) if n_obs > 0 else 0.0,
            "candidate_pair_count": len(rows),
        }
        return df_feats, diag
    return df_feats


FEATURE_COLUMNS = [
    "dist_perp_m",
    "proj_ratio_u",
    "road_length_m",
    "dist_prev_m",
    "dist_next_m",
    "u_diff_prev",
    "u_diff_next",
    "heading_diff_deg",
    "cos_heading_diff",
    "abs_cos_heading",
    "gps_speed_mps",
    "candidate_rank",
    "dist_minus_min",
    "dist_ratio",
    "heading_minus_min",
    "spatial_softmax",
    "speed_x_cos_heading",
    "candidate_count",
]
