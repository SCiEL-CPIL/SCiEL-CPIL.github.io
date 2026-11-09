"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid basis of 5 fixed-center
    Gaussian RBFs (for local features) and 5 Chebyshev polynomials (for global
    trends). This combines the local flexibility of RBFs (current program's
    strength) with the global polynomial approximation capability of Chebyshev
    (inspired by the high-scoring Chebyshev-only approach).
    
    y = sum_{i=1}^{5} p[i] * exp(-((x - c_i)^2) / (2 * sigma^2))
        + sum_{j=0}^{4} p[5+j] * T_j(x/2)
    
    where c_i are fixed centers at -2, -1, 0, 1, 2 and sigma = 0.8
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed Gaussian centers
    centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Fixed Gaussian width
    sigma = 0.8
    
    # Compute Gaussian terms: shape (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Compute Chebyshev polynomials T_0 through T_4 on t = x/2
    t = x / 2.0
    n = x.shape[0]
    cheb = np.empty((n, 5), dtype=np.float64)
    cheb[:, 0] = 1.0  # T_0
    cheb[:, 1] = t    # T_1
    
    if n > 0:
        T0 = cheb[:, 0]
        T1 = cheb[:, 1]
        for k in range(2, 5):
            Tk = 2.0 * t * T1 - T0
            cheb[:, k] = Tk
            T0, T1 = T1, Tk
    
    # Combine: first 5 params weight Gaussians, next 5 weight Chebyshev
    result = gaussians @ params[:5] + cheb @ params[5:]
    
    return result
# EVOLVE-BLOCK-END
