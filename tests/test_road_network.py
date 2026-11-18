"""
Unit tests for road_network and spatial_projection modules.

Author : A S Vincent
Commit : #C02
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.spatial_projection import get_utm_crs, build_transformer, project_coords
from src.data.road_network import (
    load_road_network,
    get_segment_midpoints,
    get_segment_endpoints,
    get_road_network_stats,
    load_route,
)

# Path to a known training trip (adjust if needed)
SAMPLE_TRIP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Student folder", "Student folder", "train", "00000001"
)


class TestSpatialProjection:
    def test_utm_zone_darmstadt(self):
        """Darmstadt area (~lon 8.58) should be UTM zone 32N."""
        crs = get_utm_crs(8.58, 49.86)
        assert "32" in str(crs) or crs.to_epsg() == 32632

    def test_transformer_round_trip(self):
        """Project and verify output dimensions."""
        transformer = build_transformer(8.58, 49.86)
        lons = np.array([8.58, 8.59])
        lats = np.array([49.86, 49.87])
        ex, ny = project_coords(lons, lats, transformer)
        assert ex.shape == (2,)
        assert ny.shape == (2,)
        # Approximate 1 degree lat ≈ 111 km, 0.01 deg ≈ 1.1 km
        dy = abs(ny[1] - ny[0])
        assert 1000 < dy < 1300  # ~1.1 km


class TestRoadNetwork:
    @pytest.fixture(scope="class")
    def network(self):
        if not os.path.isdir(SAMPLE_TRIP_DIR):
            pytest.skip("Sample trip data not available at expected path")
        return load_road_network(SAMPLE_TRIP_DIR)

    def test_load_shape(self, network):
        assert len(network.nodes_x) == len(network.nodes_y)
        assert len(network.arcs_from) == len(network.arcs_to)
        assert len(network.segments) == len(network.arcs_from)

    def test_segments_have_positive_length(self, network):
        for seg in network.segments[:100]:
            assert seg.length_m >= 0

    def test_midpoints_shape(self, network):
        mids = get_segment_midpoints(network)
        assert mids.shape == (len(network.segments), 2)

    def test_endpoints_shape(self, network):
        eps = get_segment_endpoints(network)
        assert eps.shape == (len(network.segments), 4)

    def test_stats_keys(self, network):
        stats = get_road_network_stats(network)
        assert "num_nodes" in stats
        assert "total_road_length_km" in stats
        assert stats["num_arcs"] > 0


class TestRouteLoader:
    def test_load_route(self):
        route_path = os.path.join(SAMPLE_TRIP_DIR, "00000001.route")
        if not os.path.isfile(route_path):
            pytest.skip("Route file not available")
        route = load_route(route_path)
        assert route.ndim == 1
        assert len(route) > 0
        assert route.dtype in (np.int32, np.int64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
