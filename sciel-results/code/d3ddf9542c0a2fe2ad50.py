"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis of 10 terms.
    
    The basis is designed to be well-conditioned and cover a wide range of
    possible target functions:
    - Chebyshev polynomials T_0 through T_7 (8 terms) for polynomial-like behavior
    - sin(x) and cos(x) for periodic behavior
    
    The domain is [-3, 3]. Chebyshev polynomials are evaluated on the mapped
    variable t = x/3 in [-1, 1] for numerical stability.
    
    The 10 parameters are fitted by the evaluator's BFGS optimizer.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_7(t)
    T0 = np.ones_like(t_safe)
    T1 = t_safe.copy()
    
    terms = [T0, T1]
    
    for k in range(2, 8):
        Tk = 2.0 * t_safe * terms[k-1] - terms[k-2]
        terms.append(Tk)
    
    # Add trigonometric terms
    terms.append(np.sin(x))
    terms.append(np.cos(x))
    
    # Linear combination of 10 terms
    result = np.zeros_like(t_safe)
    for i in range(10):
        result += params[i] * terms[i]
        
    # Ensure the result is finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
