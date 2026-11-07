"""
Symbolic regression with a rich basis of 10 features.
We use a linear model: y = sum_i params[i] * phi_i(x)
The basis functions are chosen to span a wide variety of common functions:
- Constant
- x, x^2, x^3, x^4, x^5 (polynomials)
- sin(x), cos(x)
- exp(-x^2) (Gaussian)
- tanh(x)

This is a linear model in the parameters, so BFGS should converge well.
The key is to choose basis functions that can represent the target well.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute y = sum_i params[i] * phi_i(x)
    
    Basis functions:
    phi_0 = 1
    phi_1 = x
    phi_2 = x^2
    phi_3 = x^3
    phi_4 = x^4
    phi_5 = x^5
    phi_6 = sin(x)
    phi_7 = cos(x)
    phi_8 = exp(-x^2)
    phi_9 = tanh(x)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure we have exactly 10 params
    if params.shape[0] != 10:
        # Pad or truncate to 10
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    # Compute basis functions
    phi = np.ones((10, x.shape[0]))
    phi[0, :] = 1.0
    phi[1, :] = x
    phi[2, :] = x ** 2
    phi[3, :] = x ** 3
    phi[4, :] = x ** 4
    phi[5, :] = x ** 5
    phi[6, :] = np.sin(x)
    phi[7, :] = np.cos(x)
    phi[8, :] = np.exp(-x ** 2)
    phi[9, :] = np.tanh(x)
    
    # y = params @ phi
    y = params @ phi
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        # Replace non-finite values with 0
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
