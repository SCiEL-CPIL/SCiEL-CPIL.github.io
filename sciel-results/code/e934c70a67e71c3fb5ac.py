"""
Strategy: Linearly Parameterized Gaussian Basis Functions (RBF) in 3D Substrate/Temp/pH space.

Rationale:
The previous mechanistic model (Monod * Gaussian * Quadratic) created a highly non-convex
loss landscape for the 10 parameters, causing BFGS to converge to poor local minima (score ~40).
This new approach uses a linear combination of basis functions:
y = sum( w_i * phi_i(x) ) + density_correction

The basis functions are Gaussian kernels centered at fixed, biologically plausible points
in the (temp, pH, s) space. The parameters are:
- 4 amplitudes (w_1..w_4) for the 4 Gaussian centers.
- 3 width parameters (sigma_t, sigma_p, sigma_s) shared across all centers to ensure stability.
- 3 coefficients for a linear density correction (1 + c1*b + c2*b^2 + c3*b^3) to handle
  population density effects without adding non-convex multiplicative coupling with the
  main growth terms.

This structure is linear in the parameters (amplitudes and density coeffs) for a fixed
width, but the widths are also optimized. However, the separation of "shape" (widths) and
"activation" (amplitudes) and the convexity of the amplitude part helps the optimizer.
Crucially, the basis functions are localized, allowing the model to capture complex,
multi-modal interactions between temperature, pH, and substrate without assuming a single
global peak or simple polynomial trends.
"""

import numpy as np
import time

# Fixed centers for the 4 Gaussian basis functions in normalized space [0, 1]
# We choose centers that cover the likely optimal region and boundaries.
# Temp: optimal ~37C. Range 1-40. (37-1)/39 = 0.923. Let's use 0.9.
# pH: optimal ~7. Range 2-12. (7-2)/10 = 0.5.
# Substrate: Monod saturation. Range 1-5. Normalized 0-1.
# Let's pick 4 centers:
# 1. Optimal: t=0.9, ph=0.5, s=0.5
# 2. High S, Optimal T/pH: t=0.9, ph=0.5, s=0.9
# 3. Lower T, Optimal pH, Med S: t=0.6, ph=0.5, s=0.5
# 4. Different pH, Optimal T, Med S: t=0.9, ph=0.7, s=0.5

# These are hard-coded to provide a good initial geometry. The optimizer will adjust
# amplitudes and widths.

CENTERS = np.array([
    [0.9, 0.5, 0.5],
    [0.9, 0.5, 0.9],
    [0.6, 0.5, 0.5],
    [0.9, 0.7, 0.5]
])

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using linearly parameterized Gaussian RBFs.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Params (length 10):
    p[0:4]: Amplitudes for the 4 Gaussian centers
    p[4]: Width for temperature dimension
    p[5]: Width for pH dimension
    p[6]: Width for substrate dimension
    p[7]: Density linear coefficient
    p[8]: Density quadratic coefficient
    p[9]: Density cubic coefficient
    """
    # Normalize inputs to [0, 1]
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    p = params
    
    # Ensure widths are positive and not too small to avoid numerical instability
    # Add a small epsilon to prevent division by zero in the Gaussian exponent
    sigma_t = np.abs(p[4]) + 1e-3
    sigma_p = np.abs(p[5]) + 1e-3
    sigma_s = np.abs(p[6]) + 1e-3
    
    # Calculate squared distances from each point to the 4 centers
    # t_n, ph_n, s_n are vectors of shape (N,)
    # CENTERS is shape (4, 3)
    
    # Distances in t dimension
    dt = t_n[:, np.newaxis] - CENTERS[:, 0][np.newaxis, :]  # Shape (N, 4)
    # Distances in pH dimension
    dph = ph_n[:, np.newaxis] - CENTERS[:, 1][np.newaxis, :]  # Shape (N, 4)
    # Distances in s dimension
    ds = s_n[:, np.newaxis] - CENTERS[:, 2][np.newaxis, :]  # Shape (N, 4)
    
    # Squared normalized distances
    d2_t = (dt / sigma_t) ** 2
    d2_p = (dph / sigma_p) ** 2
    d2_s = (ds / sigma_s) ** 2
    
    # Total squared distance
    d2_total = d2_t + d2_p + d2_s
    
    # Gaussian basis values: exp(-0.5 * d2)
    # Shape (N, 4)
    basis_vals = np.exp(-0.5 * d2_total)
    
    # Weighted sum of basis functions
    # p[0:4] are the amplitudes
    amplitudes = p[0:4]
    growth_term = basis_vals @ amplitudes  # Shape (N,)
    
    # Density correction: Linear in params, multiplicative in b_n
    # We use a polynomial in b_n to allow for inhibition at high density
    # 1 + p[7]*b_n + p[8]*b_n^2 + p[9]*b_n^3
    density_term = 1.0 + p[7] * b_n + p[8] * b_n**2 + p[9] * b_n**3
    
    # Combine
    result = growth_term * density_term
    
    return result
