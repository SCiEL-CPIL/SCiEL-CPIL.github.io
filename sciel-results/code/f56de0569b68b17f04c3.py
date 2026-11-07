"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1]
    u = x / 3.0
    
    # Hybrid basis: Chebyshev polynomials for good conditioning +
    # non-polynomial terms for richer expressiveness.
    # This keeps the well-conditioned Chebyshev structure while adding
    # flexibility to capture non-polynomial target functions.
    
    # Chebyshev polynomials T0-T6 (7 terms)
    T0 = np.ones_like(u)
    T1 = u
    T2 = 2.0 * u * T1 - T0
    T3 = 2.0 * u * T2 - T1
    T4 = 2.0 * u * T3 - T2
    T5 = 2.0 * u * T4 - T3
    T6 = 2.0 * u * T5 - T4
    
    # Non-polynomial terms (3 terms): scaled to have O(1) magnitude on [-1,1]
    # sin(pi*u) captures one full oscillation
    S1 = np.sin(np.pi * u)
    # cos(pi*u) captures the complementary oscillation  
    C1 = np.cos(np.pi * u)
    # tanh(2*u) captures sigmoid-like behavior, well-behaved on [-1,1]
    TH = np.tanh(2.0 * u)
    
    # All 10 params weight the hybrid basis.
    # Chebyshev terms provide polynomial approximation with good conditioning.
    # Trig and tanh terms add non-polynomial flexibility.
    y = (params[0] * T0 + 
         params[1] * T1 + 
         params[2] * T2 + 
         params[3] * T3 + 
         params[4] * T4 + 
         params[5] * T5 + 
         params[6] * T6 +
         params[7] * S1 +
         params[8] * C1 +
         params[9] * TH)
    
    return y
# EVOLVE-BLOCK-END
