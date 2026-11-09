"""Symbolic regression with 1 input(s) using piecewise linear hat functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a piecewise linear basis (hat functions)
    with 10 fixed knots evenly spaced across [0.1, 6].
    
    The basis functions are continuous but non-smooth, allowing the model
    to approximate sharp transitions or discontinuities that global
    polynomial/smooth bases fail to capture.
    
    Linear in parameters for stable BFGS convergence.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = np.asarray(params, dtype=np.float64)
    
    # 10 knots evenly spaced across [0.1, 6]
    # We need 10 basis functions (one per parameter)
    # Using hat functions centered at each knot
    
    # Knot positions: evenly spaced from 0.1 to 6
    # For 10 hat functions, we can place centers at:
    # x_i = 0.1 + (i + 0.5) * (6 - 0.1) / 10 for i = 0..9
    # This gives centers in the middle of each sub-interval
    
    n_knots = 10
    x_min = 0.1
    x_max = 6.0
    width = (x_max - x_min) / n_knots
    
    # Centers of each hat function
    centers = x_min + (np.arange(n_knots) + 0.5) * width
    
    # For hat functions, we need to define the support
    # A standard hat function centered at c with half-width h:
    # h(x) = max(0, 1 - |x - c| / h)
    # But we need to handle boundaries properly
    
    # Let's use a different approach: piecewise linear interpolation basis
    # Define 11 knot points (including boundaries) and 10 hat functions
    # between them
    
    # Knots: 11 points from 0.1 to 6.0
    knots = np.linspace(x_min, x_max, n_knots + 1)
    
    # Hat function i is non-zero on [knots[i], knots[i+2]]
    # with peak 1 at knots[i+1]
    # For i = 0..8 (we have 9 interior knots for 10 hat functions)
    
    # Actually, let's use 10 hat functions with centers at:
    # knots[1] through knots[10] (the 10 interior/right knots)
    # Wait, we have 11 knots (indices 0-10), so 10 intervals
    # Let's place hat centers at knots[1] through knots[10]
    
    # Simpler approach: use 10 hat functions centered at evenly spaced points
    # with appropriate support
    
    pred = np.zeros_like(x)
    
    for i in range(n_knots):
        c = centers[i]
        # Half-width of the hat function
        # Adjacent centers are 'width' apart
        # Support should extend from c - width/2 to c + width/2
        # But to make them form a partition of unity, we need:
        # Support from c - width/2 to c + width/2, with linear taper
        
        half_width = width / 2.0
        dist = np.abs(x - c)
        
        # Hat function: 1 at center, linearly decreasing to 0 at distance = width
        # Actually, for a proper hat function that forms a partition of unity:
        # h_i(x) = max(0, 1 - |x - c_i| / width)
        # This gives support of width 2*width around each center
        
        h = np.maximum(0.0, 1.0 - dist / width)
        pred += p[i] * h
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
