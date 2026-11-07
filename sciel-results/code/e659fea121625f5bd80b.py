"""Rewritten program for symbolic regression with a sum of rational basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of two rational basis functions.
    
    f(x, y) = (p0 + p1*x + p2*y) / (1 + a3*x + a4*y) 
             + (p5 + p6*x + p7*y) / (1 + a8*x + a9*y)
             
    where a_i = 0.5 * tanh(p_i) for i in {3, 4, 8, 9}.
    This keeps denominators in [1 - 0.5*sqrt(2), 1 + 0.5*sqrt(2)] approx [0.3, 1.7] on [-1,1]^2,
    ensuring stability.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    p = params

    # Reparameterize denominator coefficients for stability
    a3 = 0.5 * np.tanh(p[3])
    a4 = 0.5 * np.tanh(p[4])
    a8 = 0.5 * np.tanh(p[8])
    a9 = 0.5 * np.tanh(p[9])

    # Denominators
    den1 = 1.0 + a3 * x + a4 * y
    den2 = 1.0 + a8 * x + a9 * y

    # Safety clip for denominators to avoid division by zero or huge values
    # Min possible value for den: 1 - 0.5*1 - 0.5*1 = 0.0 (if x=y=1, a=0.5). 
    # Actually min is 1 - 0.5*1 - 0.5*1 = 0. 
    # We need to ensure it doesn't get too close to 0.
    # Clip to [0.1, 10.0]
    den1 = np.clip(den1, 0.1, 10.0)
    den2 = np.clip(den2, 0.1, 10.0)

    # Numerators
    num1 = p[0] + p[1] * x + p[2] * y
    num2 = p[5] + p[6] * x + p[7] * y

    pred = num1 / den1 + num2 / den2

    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
