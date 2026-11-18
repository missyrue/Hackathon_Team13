"""
Spatial Projection Utilities
-----------------------------
Convert WGS84 (EPSG:4326) longitude/latitude coordinates to a local
UTM projected coordinate reference system in metres, so that Euclidean
distances are physically meaningful.

Author : A S Vincent
Commit : #C02
"""

import numpy as np
from pyproj import Transformer, CRS


def _utm_zone_from_lon(lon: float) -> int:
    """Return the UTM zone number for a given longitude."""
    return int((lon + 180) / 6) + 1


def get_utm_crs(lon: float, lat: float) -> CRS:
    """
    Determine the appropriate UTM CRS for coordinates near (lon, lat).

    Parameters
    ----------
    lon : float   – Representative longitude (degrees).
    lat : float   – Representative latitude (degrees).

    Returns
    -------
    pyproj.CRS – The UTM CRS object.
    """
    zone = _utm_zone_from_lon(lon)
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    return CRS.from_epsg(epsg)


def build_transformer(lon: float, lat: float) -> Transformer:
    """
    Build a pyproj Transformer from WGS84 → UTM for the region containing
    the given representative point.

    Parameters
    ----------
    lon, lat : float – A representative point (e.g. centroid of a trip).

    Returns
    -------
    pyproj.Transformer – ready to call  transformer.transform(lat, lon)
                          returning (easting, northing) in metres.
    """
    utm_crs = get_utm_crs(lon, lat)
    return Transformer.from_crs(
        CRS.from_epsg(4326),   # WGS84
        utm_crs,
        always_xy=False,       # input order: (lat, lon) → (easting, northing)
    )


def project_coords(lons: np.ndarray, lats: np.ndarray,
                    transformer: Transformer) -> tuple[np.ndarray, np.ndarray]:
    """
    Project arrays of WGS84 (lon, lat) to UTM (easting, northing) in metres.

    Parameters
    ----------
    lons : array of float – Longitudes (degrees).
    lats : array of float – Latitudes (degrees).
    transformer : pyproj.Transformer – From `build_transformer`.

    Returns
    -------
    (eastings, northings) : tuple of np.ndarray – Coordinates in metres.
    """
    eastings, northings = transformer.transform(lats, lons)
    return np.asarray(eastings), np.asarray(northings)
