"""
Road Network Graph Parser
--------------------------
Loads .nodes and .arcs files, projects them to UTM metric coordinates,
computes road segment geometries (length, bearing, midpoint), and creates
a directed graph for routing and map matching.
"""

from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import networkx as nx

from src.data.spatial_projection import build_transformer, project_coords


@dataclass
class RoadSegment:
    arc_id: int
    from_node: int
    to_node: int
    x_from: float
    y_from: float
    x_to: float
    y_to: float
    length_m: float
    heading_rad: float
    mx: float
    my: float


@dataclass
class RoadNetwork:
    trip_id: str
    nodes_lon: np.ndarray
    nodes_lat: np.ndarray
    nodes_x: np.ndarray
    nodes_y: np.ndarray
    arcs_from: np.ndarray
    arcs_to: np.ndarray
    segments: list[RoadSegment]
    graph: nx.DiGraph


def load_route(filepath: str | Path) -> np.ndarray:
    """Loads reference .route arc sequence."""
    return np.loadtxt(filepath, dtype=np.int64).ravel()


def load_road_network(trip_dir: str | Path, trip_id: Optional[str] = None) -> RoadNetwork:
    """Loads and projects the road network for a given trip folder."""
    trip_dir = Path(trip_dir)
    if trip_id is None:
        trip_id = trip_dir.name

    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"

    nodes_data = np.loadtxt(nodes_file, dtype=np.float64)
    lons = nodes_data[:, 0]
    lats = nodes_data[:, 1]

    arcs_data = np.loadtxt(arcs_file, dtype=np.int64)
    arcs_from = arcs_data[:, 0]
    arcs_to = arcs_data[:, 1]

    # Metric projection
    tf = build_transformer(float(np.mean(lons)), float(np.mean(lats)))
    nodes_x, nodes_y = project_coords(lons, lats, tf)

    # Segment geometries
    segments: list[RoadSegment] = []
    G = nx.DiGraph()

    for arc_id in range(len(arcs_from)):
        fn = int(arcs_from[arc_id])
        tn = int(arcs_to[arc_id])
        x0, y0 = float(nodes_x[fn]), float(nodes_y[fn])
        x1, y1 = float(nodes_x[tn]), float(nodes_y[tn])

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.hypot(dx, dy))
        heading = float(np.arctan2(dy, dx))

        seg = RoadSegment(
            arc_id=arc_id,
            from_node=fn,
            to_node=tn,
            x_from=x0,
            y_from=y0,
            x_to=x1,
            y_to=y1,
            length_m=length,
            heading_rad=heading,
            mx=(x0 + x1) / 2.0,
            my=(y0 + y1) / 2.0,
        )
        segments.append(seg)
        G.add_edge(fn, tn, arc_id=arc_id, weight=length)

    return RoadNetwork(
        trip_id=trip_id,
        nodes_lon=lons,
        nodes_lat=lats,
        nodes_x=nodes_x,
        nodes_y=nodes_y,
        arcs_from=arcs_from,
        arcs_to=arcs_to,
        segments=segments,
        graph=G,
    )


def get_road_network_stats(network: RoadNetwork) -> dict:
    """Computes summary statistics of the network."""
    lengths = [s.length_m for s in network.segments]
    return {
        "trip_id": network.trip_id,
        "num_nodes": len(network.nodes_x),
        "num_arcs": len(network.segments),
        "total_road_length_km": float(np.sum(lengths)) / 1000.0,
        "avg_segment_length_m": float(np.mean(lengths)),
        "connected_components": nx.number_weakly_connected_components(network.graph),
    }
