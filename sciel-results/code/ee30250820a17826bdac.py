"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function f(x, y) using a fixed basis of 10 features.
    The basis includes linear, quadratic, cubic, and trigonometric terms.
    
    Features:
    1. 1 (bias)
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. x^3
    8. y^3
    9. sin(x)
    10. sin(y)
    """
    # Ensure inputs are 1D arrays for consistent indexing
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(x)
    
    # Build the feature matrix (n x 10)
    # Using np.column_stack for efficiency
    features = np.column_stack((
        np.ones(n),
        x,
        y,
        x**2,
        y**2,
        x * y,
        x**3,
        y**3,
        np.sin(x),
        np.sin(y)
    ))
    
    # Linear combination: features (n x 10) dot params (10,) -> (n,)
    pred = features.dot(params)
    
    # Safety check: if any prediction is non-finite, return zeros to avoid crashing the evaluator
    if not np.all(np.isfinite(pred)):
        return np.zeros(n)
        
    return pred
# EVOLVE-BLOCK-END
