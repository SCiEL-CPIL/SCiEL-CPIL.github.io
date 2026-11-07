"""Deterministic Fixed-Geometry RBF Portfolio:
Uses 10 parameters to weight a fixed set of RBFs (Halton-distributed centers, varied widths) + bias.
BFGS optimizes the weights directly, avoiding internal search loops and ensuring stability.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Wall-clock deadline guard
_T0 = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _halton_sequence(count, base):
    """Generate Halton sequence in [0, 1)."""
    seq = []
    for i in range(1, count + 1):
        f = 1.0
        r = 0.0
        n = i
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        seq.append(r)
    return np.array(seq)

def _generate_rbf_centers_and_widths(n_centers=8):
    """
    Generate deterministic RBF centers and widths.
    Centers: Halton sequences mapped to [-1, 1].
    Widths: Varied to capture different frequency scales.
    """
    # Halton bases for 2D
    h1 = _halton_sequence(n_centers, 2)
    h2 = _halton_sequence(n_centers, 3)
    
    # Map to [-1, 1]
    cx = 2.0 * h1 - 1.0
    cy = 2.0 * h2 - 1.0
    
    # Varied widths: mix of narrow and wide
    # Use a deterministic pattern of widths
    widths = np.array([0.3, 0.5, 0.8, 1.2, 0.4, 0.6, 1.0, 1.5])[:n_centers]
    
    return np.column_stack([cx, cy]), widths

# Pre-compute the RBF centers and widths (deterministic, no randomness)
# We need 8 RBFs + 1 bias = 9 parameters. The 10th can be a global scale or another feature.
# Let's use 8 RBFs + 1 linear x + 1 linear y? No, let's stick to RBFs + bias for simplicity and stability.
# Actually, 10 params: 8 RBF weights + 1 bias + 1 "global variance" scale? 
# Or better: 9 RBFs + 1 bias.
N_RBF = 9
CENTERS, WIDTHS = _generate_rbf_centers_and_widths(N_RBF)

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x, y: input arrays (or a single array of shape (N, 2)? The contract says equation(x, y, params)).
    Usually x and y are separate arrays of length N.
    params: array of length 10.
    Returns: predictions array of length N.
    """
    # Ensure inputs are 1D arrays
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    
    # If n is 0, return empty
    if n == 0:
        return np.array([])
    
    # params: [w1..w9, bias]
    # We assume BFGS will fit these.
    # The function must return the prediction for the given x, y using the current params.
    
    # Compute RBF features
    # Centers: (N_RBF, 2)
    # x, y: (N,)
    # We need distance from each point (x_i, y_i) to each center (cx_j, cy_j)
    
    # Efficient computation:
    # dx = x[:, None] - cx[None, :]  -> (N, N_RBF)
    # dy = y[:, None] - cy[None, :]  -> (N, N_RBF)
    
    cx = CENTERS[:, 0]
    cy = CENTERS[:, 1]
    w = WIDTHS
    
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    
    # Squared distances
    dist_sq = dx**2 + dy**2
    
    # RBF values: exp(-dist_sq / (2 * width^2))
    # widths: (N_RBF,)
    # We need to divide by 2 * w^2 for each column
    inv_2w2 = 1.0 / (2.0 * w**2)
    
    rbf_vals = np.exp(-dist_sq * inv_2w2[None, :])
    
    # Weights: params[:N_RBF]
    weights = params[:N_RBF]
    
    # Prediction: sum(weights_j * rbf_vals_ij) + bias
    bias = params[N_RBF]
    
    pred = rbf_vals @ weights + bias
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        # Fallback: return zeros or mean? 
        # The contract says non-finite error costs 1e30. 
        # But if BFGS drives params to infinity, we should clip or handle.
        # However, the evaluator handles non-finite in the loss. 
        # We should just return the value. If it's inf, the loss will be inf.
        # To be safe, clip params? No, we don't control BFGS steps.
        # Just return the raw result.
        pass
        
    return pred

# EVOLVE-BLOCK-END
