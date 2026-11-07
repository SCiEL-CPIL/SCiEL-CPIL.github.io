"""Symbolic regression with a clean, linear, well-conditioned 10-term basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover unknown function f(x,y) using a direct linear combination of 10
    well-chosen basis functions.
    
    Basis functions:
    1. Constant (1)
    2. Linear x
    3. Linear y
    4. Interaction x*y
    5. Quadratic x^2
    6. Quadratic y^2
    7. Sine x: sin(pi*x)
    8. Sine y: sin(pi*y)
    9. Cosine x: cos(pi*x)
    10. Cosine y: cos(pi*y)
    
    This model is linear in the 10 parameters, allowing BFGS to converge
    rapidly to the global minimum for this subspace. The basis is chosen
    to cover constant, linear, interaction, quadratic, and low-frequency
    periodic components.
    """
    start_time = time.time()
    
    # Build the 10 basis functions
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * y
    b5 = x**2
    b6 = y**2
    b7 = np.sin(np.pi * x)
    b8 = np.sin(np.pi * y)
    b9 = np.cos(np.pi * x)
    b10 = np.cos(np.pi * y)
    
    # Stack into matrix: shape (n_points, 10)
    B = np.column_stack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10])
    
    # Compute prediction: B @ params
    pred = B @ params
    
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return pred
        
    return pred
# EVOLVE-BLOCK-END
