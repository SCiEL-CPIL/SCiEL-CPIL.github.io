"""Product-of-sines architecture with multiple terms for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: sum over 3 sinusoidal terms.
    Each term has amplitude, x-freq, y-freq, phase: 4 params per term = 12 total.
    But we only have 10 params, so we use 2 full terms (8 params) + 1 shared-frequency term (2 params).
    
    Actually, let's use 2 full sinusoidal terms (8 params) + bias (1) + one more amplitude (1) = 10.
    
    Term 1: p0 * sin(p1*x + p2*y + p3)
    Term 2: p4 * sin(p5*x + p6*y + p7)
    Bias:   p8
    Extra:  p9 * cos(x - y)  (fixed frequency 1, captures anti-diagonal structure)
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    term1 = p0 * np.sin(p1 * x + p2 * y + p3)
    term2 = p4 * np.sin(p5 * x + p6 * y + p7)
    term3 = p9 * np.cos(x - y)
    
    return term1 + term2 + term3 + p8
# EVOLVE-BLOCK-END
