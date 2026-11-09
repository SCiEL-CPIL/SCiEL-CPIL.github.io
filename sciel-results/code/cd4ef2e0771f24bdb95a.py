"""Symbolic regression with 1 input(s) using adaptive RBF + polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time globally
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    We parameterize the model as a linear combination of basis functions.
    The evaluator will use scipy.optimize.minimize (BFGS) to fit the 10 parameters.
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: RBF centered at 1.0 with adaptive width (fixed at 0.8)
    5: RBF centered at 3.0 with adaptive width (fixed at 0.8)
    6: RBF centered at 5.0 with adaptive width (fixed at 0.8)
    7: exp(-x)
    8: log(x)
    9: 1/x
    
    The RBF centers are chosen to cover the domain [0.1, 6] roughly evenly.
    The widths are fixed to avoid making the optimization problem too complex
    for BFGS starting from all ones. However, using RBFs allows the model to
    capture local features better than a pure polynomial or global trig basis.
    
    To improve convergence, we scale the RBFs so their peak values are ~1.
    """
    # Deadline check
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        # Return a safe default: just the polynomial part with current params
        x = np.asarray(x, dtype=np.float64)
        return params[0] + params[1] * x + params[2] * x**2
    
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
    
    p = params
    
    # Polynomial bases
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    
    # RBF bases: exp(-(x-c)^2 / (2*sigma^2))
    # Centers: 1.0, 3.0, 5.0
    # Widths: fixed at 0.8 for stability
    sigma = 0.8
    c1, c2, c3 = 1.0, 3.0, 5.0
    
    b4 = np.exp(-((x - c1)**2) / (2.0 * sigma**2))
    b5 = np.exp(-((x - c2)**2) / (2.0 * sigma**2))
    b6 = np.exp(-((x - c3)**2) / (2.0 * sigma**2))
    
    # Other bases
    b7 = np.exp(-x)
    b8 = np.log(np.clip(x, 1e-10, None))
    b9 = 1.0 / np.clip(x, 1e-10, None)
    
    # Linear combination
    pred = (p[0] * b0 +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
