"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    The evaluator fits `params` (length 10) via BFGS starting from all ones,
    minimizing training MSE. We design a flexible model that can represent
    many common function families.
    
    Strategy: Use a combination of basis functions with learnable coefficients.
    The 10 parameters are allocated to:
      - params[0]: linear coefficient
      - params[1]: quadratic coefficient
      - params[2]: cubic coefficient
      - params[3]: sine frequency
      - params[4]: sine amplitude
      - params[5]: cosine frequency
      - params[6]: cosine amplitude
      - params[7]: exponential rate
      - params[8]: exponential amplitude
      - params[9]: constant offset
    
    We also add a softplus-like activation to allow non-linear mixing.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    a_lin = params[0]
    a_quad = params[1]
    a_cub = params[2]
    f_sin = params[3]
    a_sin = params[4]
    f_cos = params[5]
    a_cos = params[6]
    rate_exp = params[7]
    a_exp = params[8]
    offset = params[9]
    
    # Clip frequencies and rates to avoid numerical issues
    f_sin = np.clip(f_sin, -50, 50)
    f_cos = np.clip(f_cos, -50, 50)
    rate_exp = np.clip(rate_exp, -10, 10)
    
    # Compute basis functions
    y_lin = a_lin * x
    y_quad = a_quad * x**2
    y_cub = a_cub * x**3
    
    y_sin = a_sin * np.sin(f_sin * x)
    y_cos = a_cos * np.cos(f_cos * x)
    
    # Exponential: use np.exp with clipping for stability
    exp_arg = rate_exp * x
    exp_arg = np.clip(exp_arg, -50, 50)
    y_exp = a_exp * np.exp(exp_arg)
    
    # Sum all components
    result = y_lin + y_quad + y_cub + y_sin + y_cos + y_exp + offset
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
