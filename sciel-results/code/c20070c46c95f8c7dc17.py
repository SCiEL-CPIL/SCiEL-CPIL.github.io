"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a hybrid basis of Chebyshev polynomials
    (for global smoothness) and Gaussian RBFs (for local flexibility).
    
    Basis:
    1-5: Chebyshev Polynomials T_0 to T_4 mapped to [-1, 1]
    6-10: Gaussian RBFs centered at -3, -1.5, 0, 1.5, 3 with sigma=1.0
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev
    t = x / 3.0

    # Chebyshev Polynomials T_0 to T_4
    # T_0(t) = 1
    f0 = np.ones_like(t)
    
    # T_1(t) = t
    f1 = t
    
    # T_2(t) = 2t^2 - 1
    f2 = 2.0 * t**2 - 1.0
    
    # T_3(t) = 4t^3 - 3t
    f3 = 4.0 * t**3 - 3.0 * t
    
    # T_4(t) = 8t^4 - 8t^2 + 1
    f4 = 8.0 * t**4 - 8.0 * t**2 + 1.0

    # Gaussian RBFs centered at -3, -1.5, 0, 1.5, 3 with sigma=1.0
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    
    # Compute RBF values: exp(-(x - center)^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-diff**2 / (2.0 * sigma**2))
    
    f5 = rbf[:, 0]
    f6 = rbf[:, 1]
    f7 = rbf[:, 2]
    f8 = rbf[:, 3]
    f9 = rbf[:, 4]

    # Build feature matrix (N, 10)
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))

    # Linear combination
    result = features @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result

# EVOLVE-BLOCK-END
