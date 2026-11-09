"""Symbolic regression with 1 input(s) using a rich mixed basis with RBF components."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using piecewise linear hat functions with 10 fixed knots
    evenly spaced across [0.1, 6]. This creates a continuous but non-smooth function
    class that can approximate discontinuities or sharp transitions. Linear in parameters.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = np.asarray(params, dtype=np.float64)
    
    # Define 10 evenly spaced knots across [0.1, 6]
    x_min, x_max = 0.1, 6.0
    knots = np.linspace(x_min, x_max, 10)
    
    # Create hat functions (triangular basis functions)
    # For each knot i, the hat function is:
    # - 0 for x <= knots[i-1] or x >= knots[i+1]
    # - Linear rise from 0 to 1 between knots[i-1] and knots[i]
    # - Linear fall from 1 to 0 between knots[i] and knots[i+1]
    # Edge knots get one-sided triangles
    
    pred = np.zeros_like(x)
    
    for i in range(10):
        if i == 0:
            # Left edge: triangle from knots[0] to knots[1]
            left = knots[0]
            right = knots[1]
            # Hat = (x - left) / (right - left) for left <= x <= right, else 0
            hat = np.clip((x - left) / (right - left), 0, 1)
        elif i == 9:
            # Right edge: triangle from knots[8] to knots[9]
            left = knots[8]
            right = knots[9]
            # Hat = (right - x) / (right - left) for left <= x <= right, else 0
            hat = np.clip((right - x) / (right - left), 0, 1)
        else:
            # Interior: triangle with peak at knots[i]
            left = knots[i-1]
            center = knots[i]
            right = knots[i+1]
            # Rising part: (x - left) / (center - left) for left <= x <= center
            # Falling part: (right - x) / (right - center) for center <= x <= right
            hat = np.where(
                x < center,
                np.clip((x - left) / (center - left), 0, 1),
                np.clip((right - x) / (right - center), 0, 1)
            )
        
        pred += p[i] * hat
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
