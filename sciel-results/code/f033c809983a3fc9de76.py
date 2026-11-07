"""Hybrid basis: sigmoids + polynomial for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: sum of shifted sigmoids + low-order polynomial.
    
    y = a0 + a1*x + a2*x^2 + sum_{i=0}^{3} w_i * sigmoid(s_i * (x - c_i))
    
    params layout (length 10):
      params[0]   = a0 (constant)
      params[1]   = a1 (linear)
      params[2]   = a2 (quadratic)
      params[3]   = w_0 (sigmoid weight 0)
      params[4]   = c_0 (sigmoid center 0)
      params[5]   = w_1 (sigmoid weight 1)
      params[6]   = c_1 (sigmoid center 1)
      params[7]   = w_2 (sigmoid weight 2)
      params[8]   = c_2 (sigmoid center 2)
      params[9]   = c_3 (sigmoid center 3, with fixed weight w_3=1.0)
    
    Fixed steepness s=2.5 for all sigmoids to ensure smoothness.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    w0 = params[3]
    c0 = params[4]
    w1 = params[5]
    c1 = params[6]
    w2 = params[7]
    c2 = params[8]
    c3 = params[9]
    
    # Polynomial part
    poly = a0 + a1 * x + a2 * x * x
    
    # Sigmoid part with fixed steepness s=2.5
    s = 2.5
    # Clip centers to [-2, 2] range to keep sigmoids relevant
    c0c = np.clip(c0, -2.0, 2.0)
    c1c = np.clip(c1, -2.0, 2.0)
    c2c = np.clip(c2, -2.0, 2.0)
    c3c = np.clip(c3, -2.0, 2.0)
    
    # Compute sigmoids: 1 / (1 + exp(-s * (x - c)))
    # Use numerically stable sigmoid
    def stable_sigmoid(z):
        # For z > 0: 1/(1+exp(-z)); for z <= 0: exp(z)/(1+exp(z))
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out
    
    z0 = s * (x - c0c)
    z1 = s * (x - c1c)
    z2 = s * (x - c2c)
    z3 = s * (x - c3c)
    
    sig0 = stable_sigmoid(z0)
    sig1 = stable_sigmoid(z1)
    sig2 = stable_sigmoid(z2)
    sig3 = stable_sigmoid(z3)
    
    result = poly + w0 * sig0 + w1 * sig1 + w2 * sig2 + 1.0 * sig3
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
