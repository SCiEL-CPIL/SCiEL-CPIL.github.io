"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    RBF + polynomial hybrid basis:
    - 5 Gaussian RBFs at fixed centers {-3, -1.5, 0, 1.5, 3}
    - 5 low-order polynomial terms (1, x, x^2, x^3, x^4)
    
    This localized representation allows BFGS to independently adjust
    local features while polynomials capture global trends.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial basis for global trends
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = b2 * b2
    
    # Gaussian RBF kernels with fixed centers and bandwidth
    # Centers chosen to cover the input domain [-3, 3]
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    bandwidth = 1.5  # sigma for Gaussian kernels
    
    # Compute RBF terms: exp(-((x - center)^2) / (2 * sigma^2))
    # Use broadcasting: x is (n,), centers is (5,)
    # diff shape: (n, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_term = np.exp(-diff * diff / (2.0 * bandwidth * bandwidth))
    
    # rbf_term has shape (n, 5), so rbf_term[:, i] is the i-th RBF
    b5 = rbf_term[:, 0]
    b6 = rbf_term[:, 1]
    b7 = rbf_term[:, 2]
    b8 = rbf_term[:, 3]
    b9 = rbf_term[:, 4]
    
    # Linear combination
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
# EVOLVE-BLOCK-END
