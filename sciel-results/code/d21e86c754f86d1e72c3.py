"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # New design: Piecewise-linear (in squared distance) RBFs (Cone in dist^2)
    # Basis: phi(d^2) = max(0, R^2 - d^2)
    # This is C1 continuous, localized, and differentiable (almost everywhere) for BFGS.
    
    # Fixed centers: 6 corners of the hypercube [0.5, 2.0]^3
    # This provides good coverage. The center of the domain is covered by all 6.
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 2.0, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 0.5, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 0.5]
    ])
    
    # Support radius squared
    # Max distance in domain is ~2.6. 
    # Distance from corner to center is ~1.3.
    # R^2 = 2.0 -> R ~ 1.41. This ensures the center is covered by all RBFs.
    R_sq = 2.0
    
    # Compute squared distances
    points = np.stack([x, y, z], axis=1)  # (N, 3)
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)  # (N, 6)
    
    # Apply basis function: max(0, R_sq - dist_sq)
    # This is linear in dist_sq, hence "piecewise-linear" in the radial metric.
    rbf_vals = np.maximum(0, R_sq - dist_sq)  # (N, 6)
    
    # Parameters:
    # 0:5 -> RBF amplitudes
    # 6:9 -> Linear coefficients for x, y, z
    # 9 -> Bias
    
    p_rbf = params[0:6]
    p_lin = params[6:9]
    p_bias = params[9]
    
    pred_rbf = rbf_vals @ p_rbf
    pred_lin = p_lin[0] * x + p_lin[1] * y + p_lin[2] * z
    pred_bias = p_bias
    
    return pred_rbf + pred_lin + pred_bias
# EVOLVE-BLOCK-END
