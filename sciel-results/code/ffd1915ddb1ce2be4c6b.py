"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear RBF approximation using 10 fixed centers.
    Basis function: phi(r) = max(0, 1 - r/sigma)
    """
    # Define 10 centers covering the domain [0.5, 2.0]^3
    # 8 corners of the cube
    c1 = np.array([0.5, 0.5, 0.5])
    c2 = np.array([2.0, 0.5, 0.5])
    c3 = np.array([0.5, 2.0, 0.5])
    c4 = np.array([2.0, 2.0, 0.5])
    c5 = np.array([0.5, 0.5, 2.0])
    c6 = np.array([2.0, 0.5, 2.0])
    c7 = np.array([0.5, 2.0, 2.0])
    c8 = np.array([2.0, 2.0, 2.0])
    
    # 2 additional centers to improve coverage in the middle
    # Center of the volume
    c9 = np.array([1.25, 1.25, 1.25])
    # Another point to break symmetry or cover a specific region? 
    # Let's use a center offset to help with gradients or specific features.
    # Or just another midpoint. Let's use (1.25, 0.5, 1.25)
    c10 = np.array([1.25, 0.5, 1.25])
    
    centers = np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
    
    # Sigma controls the width of the linear kernel.
    # Domain size is 1.5. A sigma of 1.5 means the kernel reaches 0 at distance 1.5.
    # This ensures significant overlap between neighbors.
    sigma = 1.5
    
    # Prepare points
    points = np.stack([x, y, z], axis=1)  # (N, 3)
    
    # Compute Euclidean distances
    diff = points[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)  # (N, 10)
    
    # Compute piecewise-linear RBF values
    # phi(r) = max(0, 1 - r/sigma)
    rbf_vals = np.maximum(0, 1 - dist / sigma)  # (N, 10)
    
    # Combine with parameters
    # params[0:10] are the amplitudes for each RBF
    pred = rbf_vals @ params
    
    return pred
# EVOLVE-BLOCK-END
