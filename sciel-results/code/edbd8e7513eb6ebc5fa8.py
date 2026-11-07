import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a hybrid basis of polynomials, trigonometric,
    exponential, and sigmoidal functions.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Compute basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x**2
    # 4. Cubic
    b3 = x**3
    # 5. Sine
    b4 = np.sin(x)
    # 6. Cosine
    b5 = np.cos(x)
    # 7. Exponential (scaled to avoid overflow/underflow issues if range was larger, but [-2,2] is safe)
    b6 = np.exp(x)
    # 8. Inverse Exponential
    b7 = np.exp(-x)
    # 9. Hyperbolic Tangent
    b8 = np.tanh(x)
    # 10. Gaussian-like / Lorentzian
    b9 = 1.0 / (1.0 + x**2)
    
    # Stack basis functions into a matrix of shape (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Compute linear combination
    # basis shape (N, 10), params shape (10,)
    return basis @ params
