"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Portfolio of forms with softmax gating.
    Uses 10 parameters:
    - params[0:6]: weights for 6 candidate forms
    - params[6:10]: auxiliary parameters for some forms (e.g., powers, scales)
    
    Forms:
    0: linear  a*x + b  (uses p0, p6)
    1: quadratic a*x^2 + b*x + c (uses p1, p7, p8)
    2: exponential a*exp(b*x) + c (uses p2, p7, p8)
    3: log a*log(x + b) + c (uses p3, p7, p8)
    4: power a*x^b + c (uses p4, p7, p8)
    5: sine a*sin(b*x) + c (uses p5, p7, p8)
    
    Actually, to keep it simple and robust, let's use a fixed set of basis functions
    and let the 10 params be coefficients for a linear combination.
    But the inspiration says "portfolio-of-forms solver: a smooth softmax gate over 6 candidate closed-form".
    
    Let's implement a simple weighted sum of 6 forms, where the weights are derived
    from a softmax of the first 6 params, and the remaining 4 params are shared shape parameters.
    """
    # Deadline guard
    t_start = time.time()
    
    # Softmax over first 6 params to get weights
    w_raw = params[:6]
    w_max = np.max(w_raw)
    exp_w = np.exp(w_raw - w_max)
    w = exp_w / np.sum(exp_w)
    
    # Shared parameters for the forms
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Define 6 candidate forms
    # Form 0: linear
    f0 = p6 * x + p7
    # Form 1: quadratic
    f1 = p8 * x**2 + p9 * x + p6
    # Form 2: exponential
    f2 = p6 * np.exp(p7 * x) + p8
    # Form 3: log
    f3 = p6 * np.log(x + p7) + p8
    # Form 4: power
    f4 = p6 * np.power(x + 0.1, p7) + p8
    # Form 5: sine
    f5 = p6 * np.sin(p7 * x) + p8
    
    # Combine with weights
    y = w[0] * f0 + w[1] * f1 + w[2] * f2 + w[3] * f3 + w[4] * f4 + w[5] * f5
    
    # Check for non-finite values
    if not np.all(np.isfinite(y)):
        # Return a safe finite value
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
