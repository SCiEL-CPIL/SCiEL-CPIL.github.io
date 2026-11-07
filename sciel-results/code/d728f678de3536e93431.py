"""Symbolic regression with rich linear basis model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 basis functions:
    y = sum_{k=0}^{9} params[k] * phi_k(x)
    
    Basis functions chosen for expressiveness over [-2, 2]:
      phi_0(x) = 1
      phi_1(x) = x
      phi_2(x) = x^2
      phi_3(x) = x^3
      phi_4(x) = x^4
      phi_5(x) = sin(x)
      phi_6(x) = cos(x)
      phi_7(x) = exp(x)  (clipped for safety)
      phi_8(x) = exp(-x) (clipped for safety)
      phi_9(x) = x^5
    
    This is linear in params, so BFGS minimizes a convex quadratic (MSE)
    and should converge reliably to the global optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Clip x for exponential terms to prevent overflow
    x_exp = np.clip(x, -50, 50)
    
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    a4 = params[4]
    a5 = params[5]
    a6 = params[6]
    a7 = params[7]
    a8 = params[8]
    a9 = params[9]
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    result = (a0 + a1 * x + a2 * x2 + a3 * x3 + a4 * x4 + 
              a5 * np.sin(x) + a6 * np.cos(x) + 
              a7 * np.exp(x_exp) + a8 * np.exp(-x_exp) + a9 * x5)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
