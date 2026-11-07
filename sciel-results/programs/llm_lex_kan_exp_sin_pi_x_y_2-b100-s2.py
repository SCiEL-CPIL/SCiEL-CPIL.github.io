"""Rational function model with quadratic numerator and squared-positive denominator for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit

    # Ensure x and y are 1D arrays for consistent shape handling
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    # Rational function with hybrid basis:
    # Numerator: Polynomial (quadratic)
    # Denominator: 1 + (Trigonometric terms)^2  -- always positive, smooth
    #
    # Using a strictly positive, smooth squared denominator removes
    # optimization barriers for BFGS.

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute numerator (Polynomial)
    numerator = (p0 + p1 * x + p2 * y + 
                 p3 * x**2 + p4 * y**2 + p5 * x * y)

    # Compute denominator: 1 + (trig combination)^2
    # This ensures denominator >= 1, always positive and smooth
    trig_comb = (p6 * np.sin(x) + p7 * np.sin(y) + 
                 p8 * np.cos(x) + p9 * np.cos(y))
    denominator = 1.0 + trig_comb**2

    result = numerator / denominator

    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)

    return result
# EVOLVE-BLOCK-END
