"""Linear combination of fixed basis functions for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _stable_sigmoid(z):
    """Numerically stable sigmoid."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 fixed basis functions.
    
    y = sum_{i=0}^{9} w_i * phi_i(x)
    
    where phi_i are fixed basis functions. This is linear in params,
    so BFGS converges reliably to the global optimum.
    """
    x = np.asarray(x, dtype=float)
    
    # 10 fixed basis functions
    phi0 = np.ones_like(x)          # constant
    phi1 = x                         # linear
    phi2 = x * x                     # quadratic
    phi3 = phi2 * x                  # cubic
    phi4 = phi2 * phi2               # quartic
    phi5 = _stable_sigmoid(3.0 * x)  # sigmoid, moderate scale
    phi6 = _stable_sigmoid(6.0 * x)  # sigmoid, sharp scale
    phi7 = np.exp(-x * x)            # Gaussian bump
    phi8 = np.cos(2.0 * x)           # cosine oscillation
    phi9 = np.sin(2.0 * x)           # sine oscillation
    
    # Linear combination
    result = (params[0] * phi0 +
              params[1] * phi1 +
              params[2] * phi2 +
              params[3] * phi3 +
              params[4] * phi4 +
              params[5] * phi5 +
              params[6] * phi6 +
              params[7] * phi7 +
              params[8] * phi8 +
              params[9] * phi9)
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
