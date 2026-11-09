"""Symbolic regression with 1 input(s) using piecewise linear hat functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill

# Precompute knot positions: 10 evenly spaced knots across [0.1, 6]
_KNOTS = np.linspace(0.1, 6.0, 10)


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() > _DEADLINE:
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = params
    
    # Build piecewise linear hat functions for each knot
    # For each knot i, the hat function is:
    # - 0 for x <= knots[i-1] (if i > 0)
    # - linearly increases from 0 to 1 between knots[i-1] and knots[i]
    # - linearly decreases from 1 to 0 between knots[i] and knots[i+1]
    # - 0 for x >= knots[i+1] (if i < n-1)
    # Boundary knots get half-hats (only one side)
    
    n_knots = len(_KNOTS)
    pred = np.zeros_like(x)
    
    for i in range(n_knots):
        if i == 0:
            # Left boundary: linearly decreases from 1 at knot[0] to 0 at knot[1]
            if n_knots > 1:
                width = _KNOTS[1] - _KNOTS[0]
                mask = (x >= _KNOTS[0]) & (x <= _KNOTS[1])
                val = np.where(mask, 1.0 - (x - _KNOTS[0]) / width, 0.0)
                pred += p[i] * val
        elif i == n_knots - 1:
            # Right boundary: linearly increases from 0 at knot[n-2] to 1 at knot[n-1]
            width = _KNOTS[n_knots-1] - _KNOTS[n_knots-2]
            mask = (x >= _KNOTS[n_knots-2]) & (x <= _KNOTS[n_knots-1])
            val = np.where(mask, (x - _KNOTS[n_knots-2]) / width, 0.0)
            pred += p[i] * val
        else:
            # Interior knot: full hat function
            left_width = _KNOTS[i] - _KNOTS[i-1]
            right_width = _KNOTS[i+1] - _KNOTS[i]
            
            # Left side: increases from 0 to 1
            mask_left = (x >= _KNOTS[i-1]) & (x <= _KNOTS[i])
            val_left = np.where(mask_left, (x - _KNOTS[i-1]) / left_width, 0.0)
            
            # Right side: decreases from 1 to 0
            mask_right = (x >= _KNOTS[i]) & (x <= _KNOTS[i+1])
            val_right = np.where(mask_right, 1.0 - (x - _KNOTS[i]) / right_width, 0.0)
            
            pred += p[i] * (val_left + val_right)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
