"""E. coli growth rate model with Monod kinetics, Gaussian temp/pH, and density inhibition."""
import numpy as np
import time

# Deadline guard: track start time to ensure we return within limits
# The equation function itself is called many times during fitting, so we keep it fast.
# The deadline guard is primarily for any pre-computation, but since we have none,
# we just ensure the expression is numerically stable and fast.

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Structure:
    - Substrate: Monod kinetics s / (Ks + s)
    - Temperature: Gaussian centered at T_opt with width sigma_T
    - pH: Gaussian centered at pH_opt with width sigma_pH
    - Density: Haldane-type inhibition b / (1 + (b/Kb)^2) to capture saturation and inhibition
    - Overall: mu_max * f(s) * f_T(temp) * f_pH(pH) * f_b(b)
    
    Parameters (10 total):
    p0: mu_max (max growth rate)
    p1: Ks (half-saturation for substrate)
    p2: T_opt (optimal temperature)
    p3: sigma_T (temperature width)
    p4: pH_opt (optimal pH)
    p5: sigma_pH (pH width)
    p6: Kb (density half-inhibition)
    p7: exponent for density inhibition (flexibility)
    p8: offset (baseline)
    p9: additional scaling for substrate term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive values for widths and half-saturation constants
    # Use softplus-like transformation to keep values positive and differentiable
    Ks = np.exp(p1) + 1e-6
    sigma_T = np.exp(p3) + 0.5
    sigma_pH = np.exp(p5) + 0.5
    Kb = np.exp(p6) + 1e-6
    
    # Center the optima to help BFGS from initial guess of 1s
    # T range [1, 40], so optimal is around 37 for E. coli
    T_opt = 37.0 + 0.1 * p2
    # pH range [2, 12], optimal around 7
    pH_opt = 7.0 + 0.1 * p4
    
    # Monod substrate term
    monod_s = s / (Ks + s)
    
    # Temperature Gaussian
    # Use squared difference for smoothness
    T_diff = (temp - T_opt) / sigma_T
    f_T = np.exp(-0.5 * T_diff * T_diff)
    
    # pH Gaussian
    pH_diff = (pH - pH_opt) / sigma_pH
    f_pH = np.exp(-0.5 * pH_diff * pH_diff)
    
    # Density inhibition: Haldane-like
    # b / (Kb + b + b^2/(Kb*Kb)) -- but let's use a simpler form
    # f_b = b / (Kb + b + (b/Kb)^2 * Kb) = b / (Kb + b + b^2/Kb)
    # Or simpler: 1 / (1 + (b/Kb)^n) for inhibition
    # Let's use: (b / Kb) / (1 + (b / Kb)^2) which peaks at b=Kb
    b_ratio = b / Kb
    f_b = b_ratio / (1.0 + b_ratio * b_ratio)
    
    # Combine
    mu_max = np.exp(p0) + 1e-6
    scaling = np.exp(p9) + 1e-6
    
    result = mu_max * scaling * monod_s * f_T * f_pH * f_b + p8
    
    # Clip to reasonable range to avoid numerical issues
    # Target range is [-0.00277, 1.553]
    return np.clip(result, -1.0, 3.0)
