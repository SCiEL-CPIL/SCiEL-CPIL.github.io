"""Rewritten program using a multi-scale Gaussian RBF network."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network with 10 parameters as centers.
    Fixed multi-scale widths: [0.1, 0.2, 0.5, 1.0, 2.0] (repeated for 10 centers).
    Amplitudes are fixed to 1.0 for each basis function.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # 10 centers from params
    centers = params  # shape (10,)
    
    # Fixed widths: 5 scales, each used for 2 centers
    # Scales: 0.1, 0.2, 0.5, 1.0, 2.0
    # Centers 0,1 use sigma=0.1; 2,3 use sigma=0.2; 4,5 use sigma=0.5; 6,7 use sigma=1.0; 8,9 use sigma=2.0
    sigmas = np.array([0.1, 0.1, 0.2, 0.2, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
    
    # Compute RBF values: for each center k, compute exp(-((x - c_k)^2) / (2 * sigma_k^2))
    # x shape (n,), centers shape (10,)
    # diff shape (n, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 10)
    
    # sigma shape (10,)
    sigma_sq = (sigmas * sigmas)  # (10,)
    
    # rbf shape (n, 10)
    rbf = np.exp(-0.5 * (diff ** 2) / sigma_sq[np.newaxis, :])
    
    # Amplitudes all 1.0, so prediction is sum over k of rbf[:, k] * 1.0
    # But wait, we need to use params somehow. Let me reconsider.
    
    # Actually, the problem says params is length 10. If I use centers = params,
    # then the amplitudes are fixed. But this might not work well because
    # BFGS starting from all ones means all centers start at 1.0.
    
    # Alternative: use params as amplitudes, with fixed centers.
    # Let me try fixed centers spread over [-2, 2] and params as amplitudes.
    
    # Fixed centers evenly spaced in [-2, 2]
    fixed_centers = np.linspace(-2.0, 2.0, 10)
    
    # Recompute with fixed centers
    diff = x[:, np.newaxis] - fixed_centers[np.newaxis, :]  # (n, 10)
    rbf = np.exp(-0.5 * (diff ** 2) / sigma_sq[np.newaxis, :])
    
    # params as amplitudes, shape (10,)
    # prediction = rbf @ params
    pred = rbf @ params
    
    return pred
# EVOLVE-BLOCK-END
