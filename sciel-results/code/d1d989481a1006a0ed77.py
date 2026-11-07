"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Adaptive Radial Basis Function (RBF) Network
# Instead of a fixed polynomial/trig basis, we use a set of Gaussian bumps.
# This allows the model to fit local features and non-linearities that global
# polynomials might miss or struggle to approximate without high degrees.
#
# The 10 parameters are divided into:
# - 5 centers (c1..c5)
# - 5 widths (w1..w5)
# - Amplitudes are implicitly handled by the optimizer finding the best 
#   combination? No, we only have 10 params total.
# 
# Let's restructure:
# We have 10 parameters.
# If we use 5 RBFs, we need 5 amplitudes, 5 centers, 5 widths = 15 params. Too many.
#
# Alternative: Fixed centers, optimized widths and amplitudes?
# Or: Optimized centers, fixed widths?
#
# Let's try: 5 RBFs.
# Params 0-4: Amplitudes (A1..A5)
# Params 5-9: Centers (C1..C5)
# Widths are fixed to a reasonable value, e.g., sigma = 1.0.
# 
# Wait, fixed widths might be too rigid. 
# What if we use 4 RBFs?
# Params 0-3: Amplitudes
# Params 4-7: Centers
# Params 8-9: Widths (shared or specific?)
#
# Let's try a hybrid:
# 1 Polynomial term (linear + quadratic) + 4 RBFs?
# 
# Actually, the most robust "universal" approach with 10 params is often a 
# Chebyshev polynomial of degree 9, but that's what the current one is basically 
# (a mix of power basis). The issue is ill-conditioning.
#
# Let's try a different structural change: **Mixture of Gaussians** with 
# **Optimized Centers and Widths** but **Fixed Amplitudes**? No, amplitudes must be optimized.
#
# Let's go with 5 RBFs with optimized Centers and Amplitudes, and Fixed Widths.
# Fixed widths allow the optimizer to focus on placement and scaling.
# If the function is smooth, RBFs with fixed width can approximate it well.
#
# Basis:
# y = sum_{i=1 to 5} A_i * exp(-(x - C_i)^2 / (2 * sigma_i^2))
#
# We have 10 params.
# Let's use 4 RBFs to allow for 2 extra params for a global linear trend?
# y = p[0] + p[1]*x + sum_{i=1 to 4} p[2+i-1] * exp(-(x - p[6+i-1])^2 / (2*sigma^2))
# Params:
# p[0]: Constant
# p[1]: Linear coeff
# p[2], p[3], p[4], p[5]: Amplitudes of 4 RBFs
# p[6], p[7], p[8], p[9]: Centers of 4 RBFs
# Sigma fixed to 1.0.
#
# This covers global trends (linear) and local bumps (RBFs).
# This is structurally different from pure polynomial/trig.

SIGMA = 1.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Ensure finite params
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Global linear part
    result = p[0] + p[1] * x
    
    # 4 RBFs
    # Amplitudes: p[2], p[3], p[4], p[5]
    # Centers:    p[6], p[7], p[8], p[9]
    
    for i in range(4):
        amp = p[2 + i]
        center = p[6 + i]
        # Calculate RBF
        # exp(-(x - center)^2 / (2 * sigma^2))
        diff = x - center
        rbf = np.exp(-(diff**2) / (2.0 * SIGMA**2))
        result += amp * rbf
        
    return result
# EVOLVE-BLOCK-END
