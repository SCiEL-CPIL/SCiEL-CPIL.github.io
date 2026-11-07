"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n == 0:
        return np.array([])
    
    # Map x from [-3, 3] to [-1, 1] for better conditioning
    t = x / 3.0
    
    # Hybrid Basis:
    # 1-4: Chebyshev Polynomials T_0 to T_3 (better conditioned than monomials)
    # 5-6: Trigonometric functions (sin, cos)
    # 7: Exponential (scaled)
    # 8: Hyperbolic tangent
    # 9: Cauchy kernel (smoothed reciprocal)
    # 10: Smoothed absolute value

    # Chebyshev Polynomials of the first kind
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    
    b0 = np.ones(n)
    b1 = t.copy()
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    
    # Trigonometric terms
    b4 = np.sin(x)
    b5 = np.cos(x)
    
    # Exponential term (centered to avoid overflow, scaled)
    b6 = np.exp(t)
    
    # Hyperbolic tangent
    b7 = np.tanh(t)
    
    # Smoothed reciprocal: Cauchy kernel
    b8 = 1.0 / (1.0 + t**2)
    
    # Smoothed absolute value: sqrt(t^2 + eps). eps=0.01
    b9 = np.sqrt(t**2 + 0.01)
    
    # Stack into matrix (n, 10)
    basis_matrix = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    result = basis_matrix @ params
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
