"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_rbf_centers():
    """Generate 10 deterministic RBF centers over [-1, 1]^2 using a grid + Halton-like spread."""
    # Use a combination of grid points and strategic locations for good coverage
    # 3x3 grid gives 9 points, plus 1 at center for 10 total
    grid_x = np.array([-1.0, 0.0, 1.0])
    grid_y = np.array([-1.0, 0.0, 1.0])
    
    centers = []
    for gx in grid_x:
        for gy in grid_y:
            centers.append([gx, gy])
    # Add center point (already in grid) - instead add a slightly offset point
    # Actually, let's use 10 points: 4 corners, 4 edge midpoints, 2 inner points
    centers = [
        [-1.0, -1.0],  # corner
        [1.0, -1.0],   # corner
        [-1.0, 1.0],   # corner
        [1.0, 1.0],    # corner
        [0.0, -1.0],   # edge midpoint
        [0.0, 1.0],    # edge midpoint
        [-1.0, 0.0],   # edge midpoint
        [1.0, 0.0],    # edge midpoint
        [-0.5, -0.5],  # inner
        [0.5, 0.5],    # inner
    ]
    return np.array(centers)

def _get_rbf_widths():
    """Return 10 fixed widths for the RBFs, varying to capture different scales."""
    # Mix of wide (broad features) and narrow (local features)
    widths = [
        1.2,  # wide - captures broad trends
        1.2,  # wide
        1.2,  # wide
        1.2,  # wide
        0.8,  # medium
        0.8,  # medium
        0.8,  # medium
        0.8,  # medium
        0.5,  # narrow - captures local features
        0.5,  # narrow
    ]
    return np.array(widths)

# Precompute RBF centers and widths
RBF_CENTERS = _generate_rbf_centers()
RBF_WIDTHS = _get_rbf_widths()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sparse linear combination of 10 Gaussian RBFs with fixed centers and widths.
    The optimizer fits only the 10 amplitudes (params).
    
    This decouples structural complexity (fixed geometry) from fitting,
    giving BFGS a smoother, more convex optimization landscape.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    N = len(x)
    
    # Validate params
    if not np.all(np.isfinite(params)):
        return np.zeros(N)
    
    # Compute RBF basis: 10 Gaussian RBFs
    # For each point, compute exp(-||p - c_i||^2 / (2 * w_i^2)) for i=0..9
    basis = np.zeros((N, 10))
    
    for i in range(10):
        cx, cy = RBF_CENTERS[i]
        w = RBF_WIDTHS[i]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        basis[:, i] = np.exp(-dist_sq / (2.0 * w ** 2))
    
    # Predict: basis @ params
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(N)
    
    return pred
# EVOLVE-BLOCK-END
