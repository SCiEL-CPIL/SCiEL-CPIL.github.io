"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute RBF centers and width
# 10 centers evenly spaced in [-3, 3]
_CENTERS = np.linspace(-3.0, 3.0, 10)
_WIDTH = 0.6
_SQ_WIDTH = 2.0 * _WIDTH * _WIDTH

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    # The task says "Keep an EXPLICIT deadline guard". 
    # Since we don't have a global start time, we assume the evaluator 
    # calls this function many times. A simple check isn't strictly possible 
    # without a global state, but the directive asks for it. 
    # However, the function must be stateless or use global state carefully.
    # Given the 150s limit and simple math, it's unlikely to time out.
    # We will add a dummy guard to satisfy the "explicit" requirement if 
    # we had a start time, but since we don't, we rely on the simplicity.
    # Actually, the prompt says "track elapsed time". We can use a global 
    # variable to track the first call time.
    
    global _START_TIME
    if not hasattr(equation, "_START_TIME"):
        equation._START_TIME = time.time()
    
    elapsed = time.time() - equation._START_TIME
    if elapsed > 145.0:
        # Return a safe fallback if we are about to be killed
        # Just return the current linear-like approximation or zeros
        return np.zeros_like(x)

    # Compute RBF values
    # x shape: (N,)
    # _CENTERS shape: (10,)
    # We need a matrix of distances: (N, 10)
    x_expanded = x[:, np.newaxis]
    centers_expanded = _CENTERS[np.newaxis, :]
    
    # Compute squared differences
    diff = x_expanded - centers_expanded
    sq_diff = diff * diff
    
    # Compute RBF values: exp(-sq_diff / (2 * sigma^2))
    rbf_values = np.exp(-sq_diff / _SQ_WIDTH)
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # Result: (N,)
    return rbf_values.dot(params)
# EVOLVE-BLOCK-END
