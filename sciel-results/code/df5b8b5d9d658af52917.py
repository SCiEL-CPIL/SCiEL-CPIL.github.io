"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global piecewise linear "hat function" basis with fixed knots.
    
    The basis is strictly linear in params, so BFGS converges to the
    global least-squares solution. Hat functions provide flexible,
    non-oscillatory approximation that can capture arbitrary continuous
    shapes without local minima traps of nonlinear models.
    
    We use 10 hat functions with evenly spaced knots across [-2, 2].
    Each hat is a triangular bump centered at a knot, with support
    spanning from the previous knot to the next knot.
    
    params[0:10] -> weights for 10 hat functions
    """
    # Define 10 evenly spaced knots across [-2, 2]
    # Knots at: -2, -1.5556, -1.1111, -0.6667, -0.2222, 0.2222, 0.6667, 1.1111, 1.5556, 2
    knots = np.linspace(-2.0, 2.0, 10)
    
    # For each knot, create a hat function that is 1 at the knot
    # and linearly decreases to 0 at the adjacent knots.
    # Boundary knots get one-sided triangles.
    
    result = np.zeros_like(x)
    
    for i in range(10):
        center = knots[i]
        
        if i == 0:
            # Left boundary: hat is 1 at knots[0], 0 at knots[1]
            left = knots[0]
            right = knots[1]
            # Linear from 1 at left to 0 at right
            if right > left:
                hat = np.clip((right - x) / (right - left), 0, 1)
            else:
                hat = np.zeros_like(x)
        elif i == 9:
            # Right boundary: hat is 1 at knots[9], 0 at knots[8]
            left = knots[8]
            right = knots[9]
            # Linear from 0 at left to 1 at right
            if right > left:
                hat = np.clip((x - left) / (right - left), 0, 1)
            else:
                hat = np.zeros_like(x)
        else:
            # Interior: hat is 1 at knots[i], 0 at knots[i-1] and knots[i+1]
            left = knots[i - 1]
            right = knots[i + 1]
            
            # Left side: linear from 0 at left to 1 at center
            left_part = np.clip((x - left) / (center - left), 0, 1) if center > left else np.zeros_like(x)
            
            # Right side: linear from 1 at center to 0 at right
            right_part = np.clip((right - x) / (right - center), 0, 1) if right > center else np.zeros_like(x)
            
            # Hat is the minimum of left and right parts (triangular shape)
            hat = np.minimum(left_part, right_part)
        
        result += params[i] * hat
    
    return result
# EVOLVE-BLOCK-END
