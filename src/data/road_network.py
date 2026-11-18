"""
Road Network Graph Parser
--------------------------
Load .nodes and .arcs files for a trip and build a directed graph with
projected (UTM metre) coordinates.  Exposes segment geometry (start/end
coordinates, heading vectors, lengths, bounding boxes) needed downstream
by the candidate generator and HMM transition model.

Author : A S Vincent
Commit : #C02
"""

from __future__ import annotations

import os
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Optional

from src.data.spatial_projection import build_transformer, project_coords


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class RoadSegment:
    """Geometry of one directed road arc."""
    arc_id: int
    from_node: int
    to_node: int
    # UTM projected coordinates (metres)
    x_from: float
    y_from: float
    x_to: float
    y_to: float
    # Derived properties
    length_m: float          # Euclidean length in metres
    heading_rad: float       # Bearing from start to end (radians, 0 = East, CCW+)
    # Midpoint (for spatial index)
    mx: float
    my: float
    # Bounding box (for quick spatial filtering)
    bbox_min_x: float
    bbox_min_y: float
    bbox_max_x: float
    bbox_max_y: float


@dataclass
class RoadNetwork:
    """Complete road network for one trip."""
    trip_id: str
    nodes_lon: np.ndarray        # shape (N,)  original WGS84
    nodes_lat: np.ndarray        # shape (N,)
    nodes_x: np.ndarray          # shape (N,)  UTM easting  (metres)
    nodes_y: np.ndarray          # shape (N,)  UTM northing (metres)
    arcs_from: np.ndarray        # shape (A,)  int — from_node_id per arc
    arcs_to: np.ndarray          # shape (A,)  int — to_node_id per arc
    segments: list[RoadSegment]  # length A
    graph: nx.DiGraph            # NetworkX directed graph


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_nodes(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .nodes file.

    Returns (longitudes, latitudes) arrays.  Row index = node_id.
    """
    data = np.loadtxt(filepath, dtype=np.float64)
    lons = data[:, 0]
    lats = data[:, 1]
    return lons, lats


def _load_arcs(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .arcs file.

    Returns (from_ids, to_ids) arrays.  Row index = arc_id.
    """
    data = np.loadtxt(filepath, dtype=np.int64)
    from_ids = data[:, 0]
    to_ids = data[:, 1]
    return from_ids, to_ids


def load_route(filepath: str) -> np.ndarray:
    """
    Load a .route file — the ground-truth sequence of arc_ids traversed.

    Returns 1-D int array of arc_ids.
    """
    return np.loadtxt(filepath, dtype=np.int64).ravel()


# ---------------------------------------------------------------------------
# Segment geometry computation
# ---------------------------------------------------------------------------

def _build_segments(
    arcs_from: np.ndarray,
    arcs_to: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
) -> list[RoadSegment]:
    """
    For each arc, compute projected geometry: length, heading, midpoint, bbox.
    """
    segments: list[RoadSegment] = []
    for arc_id in range(len(arcs_from)):
        fn = arcs_from[arc_id]
        tn = arcs_to[arc_id]

        x0, y0 = nodes_x[fn], nodes_y[fn]
        x1, y1 = nodes_x[tn], nodes_y[tn]

        dx = x1 - x0
        dy = y1 - y0
        length = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx)       # radians, east = 0, CCW positive

        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0

        seg = RoadSegment(
            arc_id=arc_id,
            from_node=fn,
            to_node=tn,
            x_from=x0, y_from=y0,
            x_to=x1,   y_to=y1,
            length_m=length,
            heading_rad=heading,
            mx=mx, my=my,
            bbox_min_x=min(x0, x1),
            bbox_min_y=min(y0, y1),
            bbox_max_x=max(x0, x1),
            bbox_max_y=max(y0, y1),
        )
        segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# NetworkX graph
# ---------------------------------------------------------------------------

def _build_nx_graph(
    arcs_from: np.ndarray,
    arcs_to: np.ndarray,
    segments: list[RoadSegment],
) -> nx.DiGraph:
    """
    Build a directed NetworkX graph where edges carry arc_id and length_m.
    Used by the HMM transition model for shortest-path routing.
    """
    G = nx.DiGraph()
    for seg in segments:
        G.add_edge(
            seg.from_node,
            seg.to_node,
            arc_id=seg.arc_id,
            weight=seg.length_m,
        )
    return G


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_road_network(trip_dir: str, trip_id: Optional[str] = None) -> RoadNetwork:
    """
    Load and project a road network from a trip directory.

    Parameters
    ----------
    trip_dir : str
        Path to the trip folder containing <trip_id>.nodes and <trip_id>.arcs.
    trip_id : str, optional
        Base name of the trip files (auto-detected from directory name if None).

    Returns
    -------
    RoadNetwork – Complete parsed and projected network.

    Example
    -------
    >>> net = load_road_network("data/train/00000001")
    >>> net.nodes_x.shape
    (88809,)
    >>> len(net.segments)
    165790
    """
    if trip_id is None:
        trip_id = os.path.basename(trip_dir.rstrip("/\\"))

    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path  = os.path.join(trip_dir, f"{trip_id}.arcs")

    # Load raw WGS84 coordinates
    lons, lats = _load_nodes(nodes_path)
    arcs_from, arcs_to = _load_arcs(arcs_path)

    # Build UTM transformer from centroid
    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))
    transformer = build_transformer(centroid_lon, centroid_lat)

    # Project all nodes to UTM metres
    nodes_x, nodes_y = project_coords(lons, lats, transformer)

    # Compute segment geometry
    segments = _build_segments(arcs_from, arcs_to, nodes_x, nodes_y)

    # Build graph
    graph = _build_nx_graph(arcs_from, arcs_to, segments)

    return RoadNetwork(
        trip_id=trip_id,
        nodes_lon=lons,
        nodes_lat=lats,
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        arcs_from=arcs_from,
        arcs_to=arcs_to,
        segments=segments,
        graph=graph,
    )


def get_segment_endpoints(network: RoadNetwork) -> np.ndarray:
    """
    Return (A, 4) array of [x_from, y_from, x_to, y_to] in UTM metres
    for all arc segments. Useful for vectorised spatial queries.
    """
    A = len(network.segments)
    out = np.empty((A, 4), dtype=np.float64)
    for i, seg in enumerate(network.segments):
        out[i, 0] = seg.x_from
        out[i, 1] = seg.y_from
        out[i, 2] = seg.x_to
        out[i, 3] = seg.y_to
    return out


def get_segment_midpoints(network: RoadNetwork) -> np.ndarray:
    """
    Return (A, 2) array of [mx, my] midpoints in UTM metres.
    Useful as points-of-reference for KD-Tree spatial indexing.
    """
    A = len(network.segments)
    out = np.empty((A, 2), dtype=np.float64)
    for i, seg in enumerate(network.segments):
        out[i, 0] = seg.mx
        out[i, 1] = seg.my
    return out


def get_road_network_stats(network: RoadNetwork) -> dict:
    """
    Return summary statistics for a road network (useful for EDA / report).
    """
    lengths = np.array([s.length_m for s in network.segments])
    return {
        "trip_id": network.trip_id,
        "num_nodes": len(network.nodes_x),
        "num_arcs": len(network.segments),
        "num_graph_edges": network.graph.number_of_edges(),
        "num_graph_nodes": network.graph.number_of_nodes(),
        "num_connected_components": nx.number_weakly_connected_components(network.graph),
        "avg_segment_length_m": float(np.mean(lengths)),
        "median_segment_length_m": float(np.median(lengths)),
        "max_segment_length_m": float(np.max(lengths)),
        "min_segment_length_m": float(np.min(lengths)),
        "total_road_length_km": float(np.sum(lengths)) / 1000.0,
    }
