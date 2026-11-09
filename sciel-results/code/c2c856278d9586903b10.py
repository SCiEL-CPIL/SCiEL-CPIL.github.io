# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-3, 3] to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Precompute Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    
    # Rational function: numerator / denominator
    # num = p0*T0 + p1*T1 + p2*T2 + p3*T3 + p4*T4
    # den = 1 + p5*T1 + p6*T2 + p7*T3 + p8*T4 + p9*T1^2
    num = (params[0] * T0 + 
           params[1] * T1 + 
           params[2] * T2 + 
           params[3] * T3 + 
           params[4] * T4)
    
    den = (1.0 + 
           params[5] * T1 + 
           params[6] * T2 + 
           params[7] * T3 + 
           params[8] * T4 + 
           params[9] * T1 * T1)
    
    # Safe division: replace near-zero denominators
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    
    return num / den_safe
# EVOLVE-BLOCK-END
