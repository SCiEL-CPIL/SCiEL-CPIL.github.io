"""Symbolic regression with a hybrid rational-exponential-logarithmic model."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining a rational function, an exponential term, and a logarithmic term.
    
    Structure:
    y = (Num(x) / Den(x)) + p7 * exp(p8 * u) + p9 * log(x + 1)
    
    Where:
    u = (x - 3.05) / 2.95  (Standardized input in approx [-1, 1])
    Num(x) = p0 + p1*u + p2*u^2 + p3*u^3
    Den(x) = 1.0 + p4*u + p5*u^2 + p6*u^3
    
    Parameters:
    p0-p3: Numerator coefficients
    p4-p6: Denominator coefficients (intercept fixed at 1 for stability)
    p7: Exponential amplitude
    p8: Exponential rate
    p9: Logarithmic coefficient
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    
    # Ensure p has length 10
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=1.0)
    else:
        p = p[:10]
        
    # Standardize x: map [0.1, 6] to approx [-1, 1]
    # Center: (0.1 + 6) / 2 = 3.05, Scale: (6 - 0.1) / 2 = 2.95
    x_center = 3.05
    x_scale = 2.95
    u = (x - x_center) / x_scale
    
    # Clip u to prevent extreme values if x is slightly out of bounds
    u = np.clip(u, -2.0, 2.0)
    
    # Calculate Rational Part
    # Num = p0 + p1*u + p2*u^2 + p3*u^3
    u2 = u * u
    u3 = u2 * u
    
    num = p[0] + p[1] * u + p[2] * u2 + p[3] * u3
    den = 1.0 + p[4] * u + p[5] * u2 + p[6] * u3
    
    # Safety: Ensure denominator is not close to zero
    # If den is very small, replace with a small value with the same sign (or just 1e-6)
    # Using np.where to avoid division by zero warnings
    den_safe = np.where(np.abs(den) < 1e-6, 1e-6, den)
    
    rational_part = num / den_safe
    
    # Calculate Exponential Part
    # y_exp = p7 * exp(p8 * u)
    # Clip the exponent to prevent overflow
    exp_arg = p[8] * u
    exp_arg = np.clip(exp_arg, -50.0, 50.0)
    exp_part = p[7] * np.exp(exp_arg)
    
    # Calculate Logarithmic Part
    # y_log = p9 * log(x + 1)
    # x is in [0.1, 6], so x+1 is in [1.1, 7]. Safe for log.
    x_safe = np.clip(x, 1e-10, None)
    log_part = p[9] * np.log(x_safe + 1.0)
    
    # Combine
    result = rational_part + exp_part + log_part
    
    # Final safety check for non-finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
