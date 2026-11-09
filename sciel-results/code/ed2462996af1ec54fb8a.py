"""Symbolic regression with 1 input(s) - Piecewise linear hat function basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise linear hat function basis with 10 fixed knots across [-2, 2].
    
    The basis is strictly linear in params, so BFGS converges to the global
    least-squares solution. Hat functions provide flexible, non-oscillatory
    approximation that can capture arbitrary continuous shapes.
    
    Knots are at: -2, -1.5556, -1.1111, -0.6667, -0.2222, 0.2222, 0.6667, 1.1111, 1.5556, 2
    (10 evenly spaced knots, spacing = 4/9)
    
    Hat function i peaks at knot[i] with value 1, and is 0 at knot[i-1] and knot[i+1].
    For boundary knots, the hat extends only on the interior side.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    n_knots = 10
    x_min, x_max = -2.0, 2.0
    spacing = (x_max - x_min) / (n_knots - 1)
    
    # Precompute knot positions
    knots = x_min + spacing * np.arange(n_knots)
    
    n_points = len(x)
    result = np.zeros(n_points, dtype=np.float64)
    
    for i in range(n_knots):
        if i >= len(params):
            break
        p = params[i]
        if p == 0.0:
            continue
        
        if i == 0:
            # Left boundary hat: rises from 0 at knot[0] to 1 at knot[1]
            # Actually for boundary, we want it to be 1 at knot[0] and 0 at knot[1]
            # Left hat: value = 1 - (x - knot[0]) / spacing for x in [knot[0], knot[1]]
            left = knots[0]
            right = knots[1]
            val = np.clip(1.0 - (x - left) / (right - left), 0.0, 1.0)
            # Only valid for x <= right (clip handles it)
            result += p * val
            
        elif i == n_knots - 1:
            # Right boundary hat: value = 1 at knot[n-1], 0 at knot[n-2]
            left = knots[n_knots - 2]
            right = knots[n_knots - 1]
            val = np.clip((x - left) / (right - left), 0.0, 1.0)
            result += p * val
            
        else:
            # Interior hat: peaks at knot[i], zero at knot[i-1] and knot[i+1]
            left = knots[i - 1]
            center = knots[i]
            right = knots[i + 1]
            
            # Left half: rises from 0 at left to 1 at center
            left_val = np.clip((x - left) / (center - left), 0.0, 1.0)
            # Right half: falls from 1 at center to 0 at right
            right_val = np.clip(1.0 - (x - center) / (right - center), 0.0, 1.0)
            
            val = np.minimum(left_val, right_val)
            result += p * val
    
    return result
# EVOLVE-BLOCK-END
