"""Symbolic regression with 1 input; Adaptive-width RBF basis for local smooth function fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using Radial Basis Functions (Gaussian) with fixed centers and adaptive widths.
    
    y = sum_{i=0}^{9} p_i * exp(-(x - c_i)^2 / (2 * sigma_i^2))
    
    Centers c_i are evenly spaced across the domain [0.1, 6].
    Widths sigma_i are fixed but vary spatially: narrower in the middle, wider at edges.
    This allows better resolution where the function might have more structure (middle)
    while maintaining coverage at the boundaries.
    """
    x = np.asarray(x, dtype=float)
    
    # Domain
    x_min = 0.1
    x_max = 6.0
    
    # 10 centers evenly spaced
    n_centers = 10
    centers = np.linspace(x_min, x_max, n_centers)
    
    # Adaptive bandwidth:
    # Base sigma 0.5, modulated by distance from center of domain.
    # Middle centers get slightly narrower widths (more local),
    # edge centers get wider widths (more global) to cover the domain edges.
    domain_center = (x_min + x_max) / 2.0
    domain_half = (x_max - x_min) / 2.0
    
    # Normalized distance from center of domain, in [-1, 1]
    norm_dist = (centers - domain_center) / domain_half
    
    # Width modulation: sigma_i = base_sigma * (1 + 0.3 * |norm_dist|)
    # At center (norm_dist=0): sigma = 0.5
    # At edges (norm_dist=1): sigma = 0.65
    base_sigma = 0.5
    sigma = base_sigma * (1.0 + 0.3 * np.abs(norm_dist))
    
    # Compute basis matrix
    # Shape: (n_centers, N_points)
    # We want y = params @ Basis, so Basis[i, j] = phi_i(x_j)
    
    # Efficient computation
    # (x_j - c_i)^2
    diff = x[None, :] - centers[:, None]
    sq_dist = diff ** 2
    
    # Divide by 2 * sigma_i^2 for each center
    # sigma_sq is shape (n_centers,), need to broadcast over points
    sigma_sq = 2.0 * sigma ** 2  # shape (n_centers,)
    basis = np.exp(-sq_dist / sigma_sq[:, None])
    
    y = np.dot(params, basis)
    
    return y
# EVOLVE-BLOCK-END
