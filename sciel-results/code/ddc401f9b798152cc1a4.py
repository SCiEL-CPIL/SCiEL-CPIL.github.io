"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining polynomial and trigonometric features for symbolic regression.
    
    The 10 parameters are amplitudes for a diverse set of basis functions:
    p0 * 1
    p1 * x
    p2 * y
    p3 * x**2
    p4 * x*y
    p5 * y**2
    p6 * sin(x+y)
    p7 * cos(x+y)
    p8 * sin(x-y)
    p9 * cos(x-y)
    
    This basis covers both low-order polynomial trends and periodic/oscillatory
    patterns, giving BFGS a flexible search space to recover many hidden functions.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    sum_xy = x + y
    diff_xy = x - y
    
    sin_sum = np.sin(sum_xy)
    cos_sum = np.cos(sum_xy)
    sin_diff = np.sin(diff_xy)
    cos_diff = np.cos(diff_xy)
    
    # Build the basis as a weighted sum
    basis = (p0 * np.ones_like(x) +
             p1 * x +
             p2 * y +
             p3 * x**2 +
             p4 * x * y +
             p5 * y**2 +
             p6 * sin_sum +
             p7 * cos_sum +
             p8 * sin_diff +
             p9 * cos_diff)
    
    return basis
# EVOLVE-BLOCK-END
