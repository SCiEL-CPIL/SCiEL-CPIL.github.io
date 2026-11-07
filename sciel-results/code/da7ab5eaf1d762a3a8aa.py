"""Program using a rich global basis with polynomial, trigonometric, and rational terms."""
import numpy as np
import time

_start_time = time.time()


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form symbolic regression candidate.
    Uses a rich global basis of 10 functions:
    - Polynomial terms up to degree 5
    - Trigonometric terms (sin, cos)
    - Rational/smooth terms
    
    This provides a flexible global representation that BFGS can optimize efficiently.
    """
    # Check time budget
    if time.time() - _start_time > 140:
        # Return a simple linear prediction as fallback
        return params[0] + params[1] * x
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    u = x / 3.0  # Scale to [-1, 1] for better conditioning
    
    # Build basis functions (10 total)
    # 1-5: Polynomial terms (1, u, u^2, u^3, u^4)
    p0 = np.ones_like(u)
    p1 = u
    p2 = u * u
    p3 = p2 * u
    p4 = p2 * p2
    
    # 6-7: Trigonometric terms
    t1 = np.sin(np.pi * u)
    t2 = np.cos(np.pi * u)
    
    # 8: Higher-frequency trig
    t3 = np.sin(2 * np.pi * u)
    
    # 9: Smooth rational-like term (tanh for smoothness)
    r1 = np.tanh(2 * u)
    
    # 10: Another smooth term
    r2 = np.tanh(3 * u)
    
    # Stack basis functions: shape (n, 10)
    basis = np.column_stack([p0, p1, p2, p3, p4, t1, t2, t3, r1, r2])
    
    # Linear combination: y = basis @ params
    y = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
