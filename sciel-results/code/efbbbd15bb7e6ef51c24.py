"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a combination of polynomial, trigonometric,
    and exponential basis functions to recover an unknown function.
    
    The expression is designed to be flexible enough to fit a wide variety
    of functions while remaining differentiable for BFGS optimization.
    """
    # Unpack parameters for readability
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis
    poly = p0 * x + p1 * x**2 + p2 * x**3 + p3 * x**4
    
    # Trigonometric basis (periodic components)
    trig = p4 * np.sin(x) + p5 * np.cos(x)
    
    # Exponential basis (growth/decay components)
    # Clip to avoid overflow in exp
    exp_arg = np.clip(p6 * x, -10, 10)
    exp_term = p7 * np.exp(exp_arg)
    
    # Rational-like basis (smooth, non-polynomial)
    # Use 1/(1 + x^2) as a smooth bump function
    rational = p8 / (1.0 + x**2)
    
    # Logarithmic-like basis using softplus for stability
    # softplus(x) = log(1 + exp(x))
    softplus_arg = np.clip(p9 * x, -10, 10)
    softplus = np.log1p(np.exp(softplus_arg))
    
    return poly + trig + exp_term + rational + softplus
# EVOLVE-BLOCK-END
