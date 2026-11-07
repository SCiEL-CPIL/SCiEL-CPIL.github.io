"""Symbolic regression with mixed basis (polynomial, trig, exp) scaled for BFGS conditioning."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixed basis: 3 polynomial terms (scaled), 2 cosine, 2 sine, 2 exp (scaled), 1 constant.
    Total: 10 parameters.
    
    params[0:3]  -> polynomial coefficients for 1, x/3, (x/3)^2
    params[3:5]  -> cosine coefficients for cos(pi*x/3), cos(2*pi*x/3)
    params[5:7]  -> sine coefficients for sin(pi*x/3), sin(2*pi*x/3)
    params[7:9]  -> exp coefficients for exp(x/3), exp(-x/3)
    params[9]    -> constant offset
    """
    x_scaled = x / 3.0  # Scale to [-1, 1] for better conditioning
    
    # Polynomial terms
    p0 = params[0] * np.ones_like(x_scaled)
    p1 = params[1] * x_scaled
    p2 = params[2] * x_scaled ** 2
    
    # Trigonometric terms (fundamental period matches domain)
    theta = np.pi * x_scaled
    c1 = params[3] * np.cos(theta)
    c2 = params[4] * np.cos(2 * theta)
    s1 = params[5] * np.sin(theta)
    s2 = params[6] * np.sin(2 * theta)
    
    # Exponential terms (scaled to stay in reasonable range)
    e1 = params[7] * np.exp(x_scaled)
    e2 = params[8] * np.exp(-x_scaled)
    
    # Constant
    const = params[9] * np.ones_like(x_scaled)
    
    return p0 + p1 + p2 + c1 + c2 + s1 + s2 + e1 + e2 + const
# EVOLVE-BLOCK-END
