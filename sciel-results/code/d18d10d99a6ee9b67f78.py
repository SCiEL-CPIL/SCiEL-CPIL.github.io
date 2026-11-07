"""COMBINED: Fixed-Center RBF Basis for Symbolic Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _check_deadline():
    return time.time() - _start_time > _DEADLINE

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structural Change: Fixed-Center Radial Basis Function (RBF) Basis.
    
    This model uses 10 fixed Gaussian centers distributed across the normalized
    input space. The parameters are the weights for these centers.
    
    Advantages over Quadratic Polynomial:
    1. Higher expressive power: Can model complex non-linearities (e.g., bell-shaped
       temperature/pH effects, saturation kinetics) that quadratics cannot.
    2. Still linear in parameters: BFGS converges to the global minimum for the
       weights, avoiding local minima issues of non-linear RBF fitting.
    3. Smooth and differentiable: Ideal for gradient-based optimizers like BFGS.
    
    Input Ranges (from task description):
    b: [1, 5] (Substrate concentration?)
    s: [1, 40] (Another substrate or nutrient?)
    temp: [1, 12] (Temperature)
    pH: [1, 5] (pH level)
    
    Normalization: Map each input to [0, 1] to make the RBF centers scale-invariant.
    """
    
    # 1. Normalize inputs to [0, 1]
    # Using the ranges provided in the task contract
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 39.0
    t_n = (temp - 1.0) / 11.0
    p_n = (pH - 1.0) / 4.0
    
    # Clip to [0, 1] to handle slight out-of-bounds noise if any, though task says in-range
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    t_n = np.clip(t_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # 2. Define 10 Fixed Centers in the 4D normalized space
    # We want centers that cover the space well. 
    # A grid of 2x2x2x2 would be 16 points. We have 10 params.
    # Strategy: Use a combination of corners and midpoints to capture main effects and interactions.
    # Let's select 10 strategic points:
    # 4 Corners of the "main" hypercube [0,1]^4
    # 4 Face centers (midpoints of edges along each axis)
    # 2 Body centers (midpoint of the whole cube, and one offset)
    
    centers = np.array([
        # Corners
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        # Center of the cube
        [0.5, 0.5, 0.5, 0.5],
        # A point shifted towards high temperature/pH (often critical for growth)
        [0.5, 0.5, 0.8, 0.8]
    ])
    
    # 3. Compute RBF values
    # Stack normalized inputs into a matrix of shape (n, 4)
    X = np.stack([b_n, s_n, t_n, p_n], axis=-1)
    
    # Calculate squared distances from each data point to each center
    # X: (n, 1, 4)
    # centers: (1, 10, 4)
    diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs**2, axis=-1)  # Shape: (n, 10)
    
    # Gaussian RBF: exp(-dist_sq / (2 * sigma^2))
    # Sigma choice: 0.35 provides a balance between local fitting and smoothness.
    # If sigma is too small, it's overfitting/under-smoothing. If too large, it's underfitting.
    sigma = 0.35
    rbf_values = np.exp(-dist_sq / (2.0 * sigma**2))
    
    # 4. Linear combination
    # params: (10,)
    # rbf_values: (n, 10)
    pred = rbf_values @ params
    
    return pred

# EVOLVE-BLOCK-END
