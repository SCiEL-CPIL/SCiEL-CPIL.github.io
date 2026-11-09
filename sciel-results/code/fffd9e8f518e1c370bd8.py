"""
Rewritten program using a hybrid Polynomial + Gaussian RBF basis.
This structure allows the model to capture both global polynomial trends 
and local non-linear features, acting as a universal approximator for 
smooth functions.
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of polynomial and Gaussian RBF basis functions.
    
    Basis functions:
    0: 1
    1: x
    2: x^2
    3: x^3
    4-9: Gaussian RBFs centered at 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 with sigma=1.0
    
    The evaluator fits 10 parameters via BFGS starting from all ones.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial terms to capture global trends
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b3 = x ** 3
    
    # Gaussian RBF terms to capture local non-linear features
    # Centers chosen to cover the domain [0.1, 6]
    centers = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sigma = 1.0
    
    # Compute Gaussians: exp(-0.5 * ((x - c) / sigma)^2)
    # Vectorized computation for efficiency
    # x is (N,), centers is (6,) -> diff is (N, 6)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Combine linearly
    y = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3)
         
    # Add weighted sum of Gaussians
    y += gaussians @ params[4:10]
    
    return y
