"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    Uses a hybrid basis: Chebyshev polynomials (for polynomial structure)
    plus sinusoidal and exponential terms (for periodic/exponential structure).
    
    Parameters:
      params[0:4]: Chebyshev coefficients T0, T1, T2, T3 (on [-1,1])
      params[4]: sine frequency
      params[5]: sine amplitude
      params[6]: cosine frequency
      params[7]: cosine amplitude
      params[8]: exponential rate
      params[9]: exponential amplitude
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    f_sin = params[4]
    a_sin = params[5]
    f_cos = params[6]
    a_cos = params[7]
    rate_exp = params[8]
    a_exp = params[9]
    
    # Clip frequencies and rates to avoid numerical issues
    f_sin = np.clip(f_sin, -50, 50)
    f_cos = np.clip(f_cos, -50, 50)
    rate_exp = np.clip(rate_exp, -10, 10)
    
    # Map x from [-3, 3] to t in [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Chebyshev polynomials: T0=1, T1=t, T2=2t^2-1, T3=4t^3-3t
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    
    # Polynomial part via Chebyshev basis
    y_poly = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3
    
    # Sinusoidal part
    y_sin = a_sin * np.sin(f_sin * x)
    y_cos = a_cos * np.cos(f_cos * x)
    
    # Exponential part with stability clipping
    exp_arg = rate_exp * x
    exp_arg = np.clip(exp_arg, -50, 50)
    y_exp = a_exp * np.exp(exp_arg)
    
    # Sum all components
    result = y_poly + y_sin + y_cos + y_exp
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
