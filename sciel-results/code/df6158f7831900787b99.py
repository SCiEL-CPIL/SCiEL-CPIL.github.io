"""Rewritten program using a hybrid Chebyshev-Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed parameters for the basis functions
# Chebyshev polynomials are used for global structure
# Gaussian RBFs are used for local variations

# Pre-compute constants if possible, but x varies, so we compute on the fly.
# Centers for the 5 Gaussian RBFs
_RBF_CENTERS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_RBF_WIDTH = 0.8
_RBF_INV_VAR = 1.0 / (2.0 * _RBF_WIDTH**2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Time guard (though computation is fast)
    start_time = time.time()
    
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=float)
    
    # 1. Compute Chebyshev Polynomials T0 to T4
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    # T4(x) = 8x^4 - 8x^2 + 1
    
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    T4 = 8.0 * x**4 - 8.0 * x**2 + 1.0
    
    # 2. Compute Gaussian RBFs
    # G_i(x) = exp(-(x - c_i)^2 * inv_var)
    
    # x shape: (N,)
    # _RBF_CENTERS shape: (5,)
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - _RBF_CENTERS[np.newaxis, :]
    G = np.exp(-(diff**2) * _RBF_INV_VAR)
    
    # 3. Combine linear terms
    # params[0:5] correspond to T0..T4
    # params[5:10] correspond to G[:,0]..G[:,4]
    
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * T4)
         
    # Add weighted sum of RBFs
    # G shape: (N, 5), params[5:10] shape: (5,)
    y += np.dot(G, params[5:10])
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
# EVOLVE-BLOCK-END
