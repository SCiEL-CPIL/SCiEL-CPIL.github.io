# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with polynomial, trig, sigmoid, and Gaussian basis.
    
    The sigmoid term (learnable center, steepness, amplitude) captures
    step-like or saturating behaviors. Parameters are optimized via BFGS.
    
    Basis functions (10 params):
    0: Constant
    1: Linear x
    2: Quadratic x^2
    3: Cubic x^3
    4: sin(x)
    5: cos(x)
    6: Sigmoid amplitude a
    7: Sigmoid center c
    8: Sigmoid steepness k
    9: Gaussian exp(-x^2)
    """
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    exp_neg_x2 = np.exp(-x2)
    
    # Sigmoid: a / (1 + exp(-k * (x - c)))
    # Clip to avoid overflow in exp
    z = p8 * (x - p7)
    z_clipped = np.clip(z, -50, 50)
    sigmoid = p6 / (1.0 + np.exp(-z_clipped))
    
    result = (p0 * 1.0 + 
              p1 * x + 
              p2 * x2 + 
              p3 * x3 + 
              p4 * sin_x + 
              p5 * cos_x +
              sigmoid + 
              p9 * exp_neg_x2)
    
    # Deadline guard: if we're close to the limit, return what we have
    # (in practice, this single evaluation is fast, but the guard is structural)
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
