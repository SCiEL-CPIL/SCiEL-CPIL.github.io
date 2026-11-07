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
    
    # Map x from [-3, 3] to [-1, 1] for better conditioning of polynomial-like terms
    t = x / 3.0
    
    # Build a diverse basis of 10 functions that span a wide range of common hidden functions:
    # 1. Constant
    # 2. Linear in t
    # 3. Quadratic in t
    # 4. Cubic in t
    # 5. Sine
    # 6. Cosine
    # 7. Exponential (decaying/growing)
    # 8. Hyperbolic tangent
    # 9. Reciprocal (smoothed)
    # 10. Absolute value (smoothed)
    
    # Polynomial terms (scaled)
    b0 = np.ones(n)
    b1 = t.copy()
    b2 = t**2
    b3 = t**3
    
    # Trigonometric terms
    b4 = np.sin(x)
    b5 = np.cos(x)
    
    # Exponential term (centered to avoid overflow, scaled)
    # exp(x) can be large; use exp(t) which is in [exp(-1), exp(1)]
    b6 = np.exp(t)
    
    # Hyperbolic tangent
    b7 = np.tanh(t)
    
    # Smoothed reciprocal: 1/(1+|t|) or similar. Let's use 1/(1+t^2) which is smooth and bounded.
    # Actually, 1/(1+t^2) is the Cauchy distribution kernel, smooth everywhere.
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
        # Replace non-finite with large values to avoid crashes, though basis should be finite
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
