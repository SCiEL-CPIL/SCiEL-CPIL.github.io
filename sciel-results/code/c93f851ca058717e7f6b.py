"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input x from [-2, 2] to z in [-1, 1] for stable Chebyshev evaluation
    z = x / 2.0
    
    # Precompute Chebyshev polynomials T_0(z) through T_7(z)
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * T1 - T0
    T3 = 2.0 * z * T2 - T1
    T4 = 2.0 * z * T3 - T2
    T5 = 2.0 * z * T4 - T3
    T6 = 2.0 * z * T5 - T4
    T7 = 2.0 * z * T6 - T5
    
    # Add exponential basis functions to capture non-polynomial behavior
    # e^x and e^-x are scaled to keep magnitudes reasonable for BFGS
    # Using x directly in exp might be better than z if the function is defined on x
    # Let's use exp(x) and exp(-x) as they are standard.
    # To avoid overflow/underflow issues in gradients, we clip or just rely on BFGS robustness.
    # x is in [-2, 2], so exp(x) is in [0.135, 7.389]. This is safe.
    E1 = np.exp(x)
    E2 = np.exp(-x)
    
    # Combine with parameters
    # We use 8 Chebyshev terms and 2 Exponential terms.
    # This allows fitting functions like c1*exp(x) + c2*exp(-x) + polynomial parts.
    return (params[0] * T0 + params[1] * T1 + params[2] * T2 + params[3] * T3 +
            params[4] * T4 + params[5] * T5 + params[6] * T6 + params[7] * T7 +
            params[8] * E1 + params[9] * E2)
# EVOLVE-BLOCK-END
