"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use all 10 params to build a flexible nonlinear model.
    We use a mix of polynomial, trigonometric, and rational terms.
    This gives the optimizer (BFGS) more representational power
    to fit potentially complex hidden functions.
    """
    # Check time budget - if exceeded, return a simple prediction
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback: simple linear prediction
        return params[0] * x + params[1] * y + params[2]
    
    # Build a rich feature set using all 10 parameters
    # We'll create multiple basis functions and combine them
    
    # Polynomial features
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2
    
    # Trigonometric features
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    cos_x = np.cos(x)
    cos_y = np.cos(y)
    sin_xy = np.sin(x * y)
    
    # Rational/damped features
    # Use (1 + small*param) in denominator to avoid division by zero
    denom1 = 1.0 + 0.1 * params[3]
    denom2 = 1.0 + 0.1 * params[4]
    damp_x = x / denom1
    damp_y = y / denom2
    
    # Exponential decay features
    exp_neg_x2 = np.exp(-0.5 * x2)
    exp_neg_y2 = np.exp(-0.5 * y2)
    
    # Combine: use a weighted sum of diverse basis functions
    # Each param controls the weight of a different feature group
    
    # Group 1: Low-order polynomial
    poly_low = params[0] + params[1] * x + params[2] * y
    
    # Group 2: Higher-order polynomial
    poly_high = (params[3] * x2 + params[4] * y2 + params[5] * xy +
                 params[6] * x3 + params[7] * y3 + params[8] * x2y + params[9] * xy2)
    
    # Group 3: Trigonometric
    trig = (params[0] * sin_x + params[1] * sin_y + params[2] * cos_x +
            params[3] * cos_y + params[4] * sin_xy)
    
    # Group 4: Damped/rational
    damp = params[5] * damp_x + params[6] * damp_y + params[7] * damp_x * damp_y
    
    # Group 5: Gaussian/RBF-like
    rbf = (params[8] * exp_neg_x2 * exp_neg_y2 +
           params[9] * exp_neg_x2 + params[0] * exp_neg_y2)
    
    # Weight the groups and sum
    # Use fixed mixing weights to keep the optimization landscape smooth
    result = (0.3 * poly_low + 
              0.2 * poly_high + 
              0.15 * trig + 
              0.15 * damp + 
              0.2 * rbf)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
