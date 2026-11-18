"""
Spatial Projection Module
-------------------------
Converts geographic coordinates (WGS84 degrees: longitude, latitude)
into physical metric coordinates (UTM Zone 32N / EPSG:32632 in metres).
"""

from __future__ import annotations
import numpy as np

try:
    from pyproj import Transformer, CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


def get_utm_zone(lon: float) -> int:
    """Computes UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1


def build_transformer(lon: float, lat: float):
    """
    Builds a coordinate transformer for the local region.
    Default region is UTM Zone 32N (EPSG:32632).
    """
    if HAS_PYPROJ:
        zone = get_utm_zone(lon)
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=False)
    return None


def project_coords(lons: np.ndarray, lats: np.ndarray, transformer=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects longitude and latitude arrays into metric (Easting, Northing) coordinates.
    """
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    if transformer is not None:
        eastings, northings = transformer.transform(lats, lons)
        return np.asarray(eastings, dtype=np.float64), np.asarray(northings, dtype=np.float64)

    # Lightweight metric projection fallback if pyproj is absent
    mean_lat_rad = np.radians(np.mean(lats))
    r = 6371000.0  # Earth radius in metres
    x = np.radians(lons) * r * np.cos(mean_lat_rad)
    y = np.radians(lats) * r
    return x, y


if __name__ == "__main__":
    sample_lon, sample_lat = 8.581679, 49.857880
    tf = build_transformer(sample_lon, sample_lat)
    x, y = project_coords(np.array([sample_lon]), np.array([sample_lat]), tf)
    print(f"Sample projection: lon={sample_lon}, lat={sample_lat} -> X={x[0]:.2f}m, Y={y[0]:.2f}m")
