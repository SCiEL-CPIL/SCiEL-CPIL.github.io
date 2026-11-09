"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to center around expected training ranges
    # This helps the optimizer by keeping polynomial terms small near the data center
    T_c = temp - 25.0
    P_c = pH - 7.0
    B_c = b - 3.0
    
    # Monod term for substrate
    # p1 is K_s. Add a small epsilon to avoid division by zero if p1 becomes negative
    # However, BFGS might push p1 negative. 
    # s / (p1 + s) is undefined if p1 + s == 0.
    # Using s / (abs(p1) + s) would change the shape.
    # Better to use a soft form or assume p1 > 0.
    # If p1 is negative, the model breaks.
    # To be safe, we can use: s / (p1 + s + 1e-6)
    # But if p1 is -s, it blows up.
    # Let's use a formulation that is always defined: s / (exp(p1) + s)? No, p1 is fitted.
    # Let's stick to s / (p1 + s) and hope p1 stays positive.
    # Or use: s / (p1*p1 + s) ? No, that changes semantics.
    # Let's use: s / (abs(p1) + s) ? No, non-differentiable at 0.
    # Let's use: s / (p1 + s + 1.0) ? No.
    
    # Actually, the standard Monod is robust if p1 is positive.
    # To ensure p1 is positive, we could use exp, but that consumes a parameter or changes scaling.
    # Let's just use p1 + s. If it fails, it's a local minima issue.
    # A safer denominator: (p1 + s + 1e-8)
    
    monod = s / (p1 + s + 1e-8)
    
    # Polynomial terms for Temperature, pH, and Density
    # These are centered, so values are small near the data center
    term_T = 1.0 + p2 * T_c + p3 * T_c**2 + p4 * T_c**3
    term_P = 1.0 + p5 * P_c + p6 * P_c**2 + p7 * P_c**3
    term_B = 1.0 + p8 * B_c
    
    # Combine
    pred = p0 * monod * term_T * term_P * term_B + p9
    
    return pred
# EVOLVE-BLOCK-END
