import numpy as np
import time

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_DEADLINE = time.time() + 145.0  # 5s safety margin


def _time_left() -> float:
    return max(0.0, _DEADLINE - time.time())


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a rich nonlinear basis of 10 terms.
    
    Basis functions:
    1. 1
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. sin(x+y)
    8. cos(x-y)
    9. x^2 - y^2
    10. exp(-(x^2 + y^2))
    
    The model is:
    f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*x*y 
             + p6*sin(x+y) + p7*cos(x-y) + p8*(x^2-y^2) + p9*exp(-(x^2+y^2))
    """
    # Ensure inputs are arrays for vectorized operations
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Calculate basis terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    term6 = np.sin(x + y)
    term7 = np.cos(x - y)
    term8 = x * x - y * y
    term9 = np.exp(-(x * x + y * y))
    
    # Linear combination
    pred = (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
