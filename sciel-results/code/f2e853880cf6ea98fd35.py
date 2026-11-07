"""Rewritten program for symbolic regression with Chebyshev + sinusoidal + exponential basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression: recover an unknown function of 1 input.
    
    The evaluator fits `params` (length 10) via BFGS starting from all ones,
    minimizing training MSE. We design a flexible, smooth model with 10 parameters
    using a hybrid basis:
      - Chebyshev polynomial series (degrees 0-4) for smooth components
      - Learnable-frequency sinusoidal term for periodic features
      - Learnable exponential term for asymptotic features
    
    Parameter allocation (10 total):
      params[0]: Chebyshev c0 (constant)
      params[1]: Chebyshev c1 (linear)
      params[2]: Chebyshev c2 (quadratic)
      params[3]: Chebyshev c3 (cubic)
      params[4]: Chebyshev c4 (quartic)
      params[5]: amplitude of sin(w * x)
      params[6]: frequency w of sin(w * x)
      params[7]: amplitude of cos(v * x)
      params[8]: frequency v of cos(v * x)
      params[9]: amplitude of exp(r * x)
    
    We use a smooth, differentiable expression suitable for BFGS.
    Frequencies are clipped to avoid numerical issues.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    amp_sin = params[5]
    freq_sin = params[6]
    amp_cos = params[7]
    freq_cos = params[8]
    amp_exp = params[9]
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Chebyshev polynomial component
    y_cheb = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3 + c4 * T4
    
    # Clip frequencies for numerical stability
    freq_sin = np.clip(freq_sin, -20.0, 20.0)
    freq_cos = np.clip(freq_cos, -20.0, 20.0)
    
    # Sinusoidal and cosinusoidal terms
    y_sin = amp_sin * np.sin(freq_sin * x)
    y_cos = amp_cos * np.cos(freq_cos * x)
    
    # Exponential term with clipping for stability
    # Use a smooth approximation: clip the argument before exp
    # We need a rate parameter, but we only have amp_exp as params[9]
    # Let's use a fixed small rate or derive it from another param
    # Actually, let's use amp_exp as both amplitude and have a fixed rate structure
    # Better: use a sigmoid-like smooth exponential
    # Let's use: amp_exp * (exp(x) - exp(-x)) / 2 * 0.1 to keep it bounded
    # Or simpler: amp_exp * np.tanh(x) as a smooth bounded "exponential-like" term
    # Actually, let's use a proper exponential with a small fixed rate to avoid overflow
    rate = 0.5  # Fixed rate to keep exp bounded on [-3,3]
    exp_arg = rate * x
    exp_arg = np.clip(exp_arg, -15, 15)
    y_exp = amp_exp * np.exp(exp_arg)
    
    # Combine all components
    result = y_cheb + y_sin + y_cos + y_exp
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
