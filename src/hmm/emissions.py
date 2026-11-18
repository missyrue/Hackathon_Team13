"""
HMM Emission Probability Modeling
----------------------------------
Combines Gaussian spatial error distribution with machine learning candidate confidence.
"""

from __future__ import annotations
import numpy as np


class SpatialMLEmissionModel:
    """
    Computes emission likelihoods:
    P(o_t | s_t) = (1 / (sigma * sqrt(2*pi))) * exp( - d_perp^2 / (2 * sigma^2) ) * (P_ML + epsilon)
    """

    def __init__(self, sigma: float = 8.0, ml_weight: float = 0.10):
        self.sigma = float(sigma)
        self.ml_weight = float(ml_weight)
        self.log_constant = -np.log(self.sigma * np.sqrt(2.0 * np.pi))

    def compute_log_emission(self, dist_perp_m: float, ml_prob: float = 0.5) -> float:
        """Computes log P(o_t | s_t) in numerical log-space."""
        # Spatial Gaussian log likelihood
        spatial_log = self.log_constant - (dist_perp_m ** 2) / (2.0 * (self.sigma ** 2))

        # ML classifier confidence log likelihood
        ml_prob_clamped = np.clip(ml_prob, 1e-4, 1.0 - 1e-4)
        ml_log = np.log(ml_prob_clamped)

        return float((1.0 - self.ml_weight) * spatial_log + self.ml_weight * ml_log)
