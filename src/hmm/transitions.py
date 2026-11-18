"""
HMM Transition Probability Modeling
------------------------------------
Calculates transition probabilities between candidate road segments
based on road network topological shortest-path distance vs great-circle Euclidean distance.
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from src.data.road_network import RoadNetwork


class RoadNetworkTransitionModel:
    """
    Computes HMM transition probabilities:
    P(s_t | s_{t-1}) = (1 / beta) * exp( - | d_road(u, v) - d_euclid | / beta )
    Using exact Newson & Krumm road-network driving distances between projected locations.
    """

    def __init__(self, network: RoadNetwork, beta: float = 25.0, max_routing_dist_m: float = 500.0):
        self.network = network
        self.graph = network.graph
        self.segments = network.segments
        self.beta = float(beta)
        self.max_routing_dist = float(max_routing_dist_m)
        self._graph_dist_cache = {}

    def get_graph_distance(self, u: int, v: int) -> float:
        """Computes shortest topological distance between intersection nodes u and v."""
        if u == v:
            return 0.0

        key = (u, v)
        if key in self._graph_dist_cache:
            return self._graph_dist_cache[key]

        if self.graph.has_edge(u, v):
            dist = float(self.graph[u][v].get("weight", 0.0))
        else:
            try:
                dist = float(nx.shortest_path_length(self.graph, source=u, target=v, weight="weight"))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = self.max_routing_dist

        self._graph_dist_cache[key] = dist
        return dist

    def get_road_distance(
        self,
        from_arc_id: int,
        to_arc_id: int,
        u1: float = 0.5,
        u2: float = 0.5,
    ) -> float:
        """Computes exact driving distance along the road network between projected positions."""
        s1 = self.segments[from_arc_id]
        s2 = self.segments[to_arc_id]

        if from_arc_id == to_arc_id:
            return max(0.0, float((u2 - u1) * s1.length_m))

        rem1 = max(0.0, float((1.0 - u1) * s1.length_m))
        g_dist = self.get_graph_distance(s1.to_node, s2.from_node)
        ent2 = max(0.0, float(u2 * s2.length_m))
        return rem1 + g_dist + ent2

    def compute_log_transition(
        self,
        from_arc_id: int,
        to_arc_id: int,
        d_euclid_m: float,
        u1: float = 0.5,
        u2: float = 0.5,
    ) -> float:
        """Computes log P(s_t | s_{t-1})."""
        d_road = self.get_road_distance(from_arc_id, to_arc_id, u1=u1, u2=u2)
        discrepancy = abs(d_road - d_euclid_m)
        log_prob = -np.log(self.beta) - (discrepancy / self.beta)
        return float(log_prob)
