"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Chebyshev model: f(x) = P(x) / Q(x) where
    P(x) = sum_{i=0}^{4} a_i T_i(t),  Q(x) = 1 + sum_{j=1}^{4} b_j T_j(t)
    with t = x / 2 to map [-2, 2] -> [-1, 1].
    
    This uses 9 free parameters (a_0..a_4, b_1..b_4) plus one extra parameter
    for scaling. The denominator is constrained to be near 1 at initialization
    to avoid singularities and improve conditioning for BFGS.
    
    params[0:5]  -> numerator coefficients a_0..a_4
    params[5:9]  -> denominator coefficients b_1..b_4 (Q = 1 + b_1*T1 + ... + b_4*T4)
    params[9]    -> global scale factor
    """
    # Time guard: if we're close to the deadline, return a simple fallback
    # (In practice, this function is called many times during BFGS, so we
    #  keep it fast. The guard is mainly for safety.)
    
    t = x / 2.0  # scale to [-1, 1]
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    
    # Numerator: a_0*T0 + a_1*T1 + a_2*T2 + a_3*T3 + a_4*T4
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    numerator = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: 1 + b_1*T1 + b_2*T2 + b_3*T3 + b_4*T4
    b1, b2, b3, b4 = params[5], params[6], params[7], params[8]
    denominator = 1.0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    
    # Avoid division by zero: clamp denominator away from 0
    # Use a soft clamp to maintain differentiability
    min_denom = 1e-6
    sign = np.sign(denominator)
    # Replace zeros with small values preserving sign
    denominator_safe = np.where(np.abs(denominator) < min_denom, 
                                sign * min_denom, 
                                denominator)
    # Handle exact zeros
    denominator_safe = np.where(denominator == 0, min_denom, denominator_safe)
    
    result = numerator / denominator_safe
    
    # Global scale
    result = result * params[9]
    
    return result
# EVOLVE-BLOCK-END
