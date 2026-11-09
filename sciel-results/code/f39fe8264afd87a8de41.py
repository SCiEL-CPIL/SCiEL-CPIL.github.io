"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: Chebyshev polynomials (global smoothness) + RBF (local adaptation)
    # This gives the optimizer both global and local degrees of freedom.
    
    t = x / 3.0  # normalize to [-1, 1]
    
    # --- Chebyshev basis (first 5 params) ---
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    T4 = 2 * t * T3 - T2
    
    # --- RBF basis (last 5 params) ---
    # 5 fixed centers evenly spaced in [-1, 1]
    centers = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Fixed width for stability (widths are hard to optimize with BFGS from all-ones)
    # Using a moderate width that covers a good portion of the domain
    width = 0.8
    width_sq = 2.0 * width * width
    
    # RBF_i(x) = exp(-(x - c_i)^2 / width_sq)
    rbf = np.zeros_like(x)
    for i in range(5):
        rbf += np.exp(-(t - centers[i])**2 / width_sq)
    
    # Actually, we need 5 separate RBFs, not their sum. Let me redo this.
    # Each of the last 5 params controls one RBF center's weight.
    
    # Recompute: separate RBFs
    rbf_terms = np.zeros((5, len(x)))
    for i in range(5):
        rbf_terms[i] = np.exp(-(t - centers[i])**2 / width_sq)
    
    # Combine: first 5 params for Chebyshev, last 5 for RBF
    y = (params[0] * T0 +
         params[1] * T1 +
         params[2] * T2 +
         params[3] * T3 +
         params[4] * T4 +
         params[5] * rbf_terms[0] +
         params[6] * rbf_terms[1] +
         params[7] * rbf_terms[2] +
         params[8] * rbf_terms[3] +
         params[9] * rbf_terms[4])
    
    return y
# EVOLVE-BLOCK-END
