"""
Uses a mixed polynomial and trigonometric basis.
This model is linear in the parameters, allowing the external BFGS optimizer
to find the global minimum efficiently.
It captures linear, quadratic, and periodic interactions, providing a
diverse hypothesis space for the hidden function.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters for clarity
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    # Polynomial terms
    t0 = np.ones_like(x)
    t1 = x
    t2 = y
    t3 = x * x
    t4 = y * y
    
    # Trigonometric terms
    t5 = np.sin(x)
    t6 = np.cos(x)
    t7 = np.sin(y)
    t8 = np.cos(y)
    t9 = np.sin(x + y)
    
    # Linear combination
    return (p0 * t0 + 
            p1 * t1 + 
            p2 * t2 + 
            p3 * t3 + 
            p4 * t4 + 
            p5 * t5 + 
            p6 * t6 + 
            p7 * t7 + 
            p8 * t8 + 
            p9 * t9)
# EVOLVE-BLOCK-END
