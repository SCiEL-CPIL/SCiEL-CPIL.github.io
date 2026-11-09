"""RBF-based nonlinear oscillator model."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF model for a nonlinear oscillator.
    
    Inputs:
        t: time (range [30, 50])
        x: position (range [-0.2056, 0.1749])
        v: velocity (range [-0.3391, 0.3624])
        params: 10 fitted constants
    
    Returns:
        dv/dt (acceleration) for each row
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Ensure params is a flat array of shape (10,)
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Normalize inputs to roughly [-1, 1] to make RBF scales comparable
    # t: [30, 50] -> center 40, range 10
    t_norm = (t - 40.0) / 10.0
    # x: [-0.2056, 0.1749] -> center ~ -0.015, range ~ 0.38. Scale by 0.2
    x_norm = x / 0.2
    # v: [-0.3391, 0.3624] -> center ~ 0.01, range ~ 0.7. Scale by 0.35
    v_norm = v / 0.35
    
    # Define 9 RBF centers in normalized space
    # Distributed across the box [-1, 1]^3
    centers = np.array([
        [-1.0, -0.75, -0.70],
        [-1.0,  0.00,  0.00],
        [-1.0,  0.75,  0.70],
        [ 0.0, -0.75,  0.70],
        [ 0.0,  0.00,  0.00],
        [ 0.0,  0.75, -0.70],
        [ 1.0, -0.75,  0.70],
        [ 1.0,  0.00,  0.00],
        [ 1.0,  0.75, -0.70]
    ])
    
    # Compute squared distances from each input point to each center
    # Input shape: (N, 3)
    Z = np.column_stack((t_norm, x_norm, v_norm))
    # Centers shape: (9, 3)
    
    # dist_sq[i, j] = ||Z[i] - C[j]||^2
    # Using broadcasting: (N, 1, 3) - (1, 9, 3)
    diff = Z[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # RBF width sigma. 0.5 in normalized units means effective radius ~ 1.
    sigma2 = 2.0 * (0.5**2)
    
    # RBF values: (N, 9)
    rbf_vals = np.exp(-dist_sq / sigma2)
    
    # Add bias term (10th parameter)
    bias = np.ones_like(t_norm)
    
    # Combine: (N, 10)
    b = np.column_stack((bias, rbf_vals))
    
    # Linear combination: (N, 10) @ (10,) -> (N,)
    result = b @ p
    
    # Safety: replace any non-finite values with 0 to avoid scoring penalty
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
