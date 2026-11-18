"""
Viterbi Sequential Route Decoder & Continuity Evaluator
-------------------------------------------------------
Implements the Viterbi dynamic programming trellis algorithm in log-probability space
to reconstruct the globally optimal and continuous sequence of road segments.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

from src.data.road_network import RoadNetwork
from src.hmm.transitions import RoadNetworkTransitionModel
from src.hmm.emissions import SpatialMLEmissionModel


def compute_route_continuity(route_arcs: list[int], network: RoadNetwork) -> float:
    """
    Computes route continuity percentage:
    Ratio of consecutive road arcs (s_t, s_{t+1}) where either:
    1. s_t == s_{t+1} (vehicle stays on the same road segment)
    2. seg[s_t].to_node == seg[s_{t+1}].from_node (direct road connection)
    """
    if len(route_arcs) < 2:
        return 1.0

    connected = 0
    total = len(route_arcs) - 1

    for i in range(total):
        a1, a2 = route_arcs[i], route_arcs[i + 1]
        if a1 == a2:
            connected += 1
        else:
            u = network.segments[a1].to_node
            v = network.segments[a2].from_node
            if u == v or network.graph.has_edge(u, v):
                connected += 1

    return float(connected / total)


def viterbi_decode(
    obs_candidates: list[list[dict]],
    transition_model: RoadNetworkTransitionModel,
    emission_model: SpatialMLEmissionModel,
    gps_distances: list[float],
) -> list[int]:
    """
    Decodes the most likely sequence of road segment IDs (arc_id)
    using the log-space Viterbi dynamic programming algorithm.
    """
    T = len(obs_candidates)
    if T == 0:
        return []

    # Initialize Viterbi trellis at t = 0
    # V[t][i] = max log-probability of path ending at candidate i at step t
    # backpointer[t][i] = index of best predecessor candidate at step t-1
    V = [{}]
    backpointer = [{}]

    for i, cand in enumerate(obs_candidates[0]):
        log_e = emission_model.compute_log_emission(cand["dist_perp_m"], cand.get("ml_prob", 0.5))
        # Uniform initial prior
        V[0][i] = log_e
        backpointer[0][i] = -1

    # Forward dynamic programming pass
    for t in range(1, T):
        V.append({})
        backpointer.append({})
        d_euclid = gps_distances[t] if t < len(gps_distances) else 30.0

        for j, curr_cand in enumerate(obs_candidates[t]):
            log_e = emission_model.compute_log_emission(curr_cand["dist_perp_m"], curr_cand.get("ml_prob", 0.5))

            best_val = -float("inf")
            best_prev = 0

            u2 = float(curr_cand.get("proj_ratio_u", 0.5))
            for i, prev_cand in enumerate(obs_candidates[t - 1]):
                u1 = float(prev_cand.get("proj_ratio_u", 0.5))
                log_t = transition_model.compute_log_transition(
                    prev_cand["arc_id"], curr_cand["arc_id"], d_euclid, u1=u1, u2=u2
                )
                total_log_prob = V[t - 1][i] + log_t + log_e
                if total_log_prob > best_val:
                    best_val = total_log_prob
                    best_prev = i

            V[t][j] = best_val
            backpointer[t][j] = best_prev

    # Backtracking pass
    best_last_idx = max(V[T - 1], key=V[T - 1].get)
    reconstructed_path = []
    curr_idx = best_last_idx

    for t in range(T - 1, -1, -1):
        reconstructed_path.append(obs_candidates[t][curr_idx]["arc_id"])
        curr_idx = backpointer[t][curr_idx]

    reconstructed_path.reverse()
    return reconstructed_path
