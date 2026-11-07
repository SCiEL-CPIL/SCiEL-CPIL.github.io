"""
Symbolic regression: hybrid polynomial, tanh, and trigonometric basis with 10 features.
Combines low-order polynomials, bounded tanh nonlinearities with cross-interactions,
and trigonometric terms to capture a wide variety of functional forms on [-1,1]x[-1,1].
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns predictions for given x, y, params.
    
    params has length 10, corresponding to:
        1, x, y, x^2, y^2, xy, tanh(x)*tanh(y), sin(x+y), cos(x-y), tanh(x-y)
    
    This basis includes:
    - Low-order polynomial terms (1, x, y, x^2, y^2, xy) for smooth trends
    - tanh(x)*tanh(y): bounded bilinear interaction, captures saturation and coupling
    - sin(x+y), cos(x-y): periodic/coupled oscillatory behavior
    - tanh(x-y): bounded antisymmetric interaction, captures directional saturation
    """
    # Track time to ensure we stay within budget
    t_start = time.time()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute intermediate values
    tx = np.tanh(x)
    ty = np.tanh(y)
    
    # Compute predictions
    pred = (
        p0 * np.ones_like(x) +
        p1 * x +
        p2 * y +
        p3 * x**2 +
        p4 * y**2 +
        p5 * x * y +
        p6 * tx * ty +
        p7 * np.sin(x + y) +
        p8 * np.cos(x - y) +
        p9 * np.tanh(x - y)
    )
    
    # Safety margin check: if we're getting close to the 120s limit, ensure we return
    elapsed = time.time() - t_start
    if elapsed > 115:  # 5s safety margin before 120s
        # Return what we have (should be fine since computation is fast)
        pass
    
    # Ensure the output is a finite array with correct shape
    if not np.all(np.isfinite(pred)):
        # Fallback to a simple linear model if something went wrong
        pred = p0 * np.ones_like(x) + p1 * x + p2 * y
    
    return pred
# EVOLVE-BLOCK-END
