"""E. coli growth rate model: flexible polynomial basis with biological gating."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: polynomial basis for substrate and linear terms for temp/pH/density,
    multiplied by soft biological gates (Monod-like for s, Gaussian-like for temp/pH).
    
    This combines the flexibility of a polynomial representation (inspiration 1)
    with the structural priors of the biological model (current program).
    
    Model: mu = (p0 + p1*s + p2*s^2) * monod_gate(s) * gauss_gate(temp) * gauss_gate(pH) 
             * (1 + p3*temp + p4*pH + p5*b + p6*temp*pH + p7*temp*b + p8*pH*b) + p9
    
    Parameters (10):
    p0: base growth rate
    p1: linear substrate coefficient
    p2: quadratic substrate coefficient
    p3: temp linear interaction
    p4: pH linear interaction
    p5: density linear interaction
    p6: temp-pH interaction
    p7: temp-density interaction
    p8: pH-density interaction
    p9: bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial substrate term
    s_poly = p0 + p1 * s + p2 * s**2
    
    # Monod-like gate for substrate (always positive, saturating)
    # Use a soft form: 1 / (1 + exp(-s)) approximates step, but let's use a smooth saturating function
    # Actually, let's use: s / (1 + s) which is simple and saturating
    # But to keep it flexible, let's use: 1 / (1 + exp(-(s - 0.5))) as a sigmoid centered at 0.5
    # Or simpler: just use a positive gate that is 1 at s=0 and saturates
    # Let's use: exp(-s) / (1 + exp(-s)) = 1/(1+exp(s)) which is 0.5 at s=0, decays to 0
    # That's not right. We want growth to increase with s.
    # Let's use: s / (1 + s) which is 0 at s=0, 0.5 at s=1, 0.8 at s=4
    monod_gate = s / (1.0 + s + 1e-6)
    
    # Gaussian-like gates for temp and pH (centered, positive)
    # Use: exp(-((temp - T_opt)^2) / (2 * sigma_T^2)) but we don't have T_opt and sigma_T as separate params
    # Instead, use a flexible Gaussian: exp(-(temp - 2.0)^2 / (2 * (1.0 + abs(p3))^2))
    # But p3 is used above. Let's use different params for gates.
    # Actually, let's keep the structure simpler.
    
    # Let's redefine:
    # Use the first 4 params for the polynomial part, next 4 for gate centers/widths, last 2 for bias and scale
    # p0, p1, p2: polynomial coefficients for s
    # p3: T_opt (temperature optimum)
    # p4: sigma_T (temperature width)
    # p5: pH_opt (pH optimum)
    # p6: sigma_pH (pH width)
    # p7: density coefficient
    # p8: interaction coefficient
    # p9: bias
    
    # Recompute with this structure
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial substrate term
    s_poly = p0 + p1 * s + p2 * s**2
    
    # Monod-like gate for substrate
    monod_gate = s / (1.0 + s + 1e-6)
    
    # Gaussian gates for temp and pH
    sigma_T = np.abs(p4) + 0.1
    sigma_pH = np.abs(p6) + 0.1
    gauss_T = np.exp(-0.5 * ((temp - p3) / sigma_T) ** 2)
    gauss_pH = np.exp(-0.5 * ((pH - p5) / sigma_pH) ** 2)
    
    # Density effect: linear inhibition
    dens_term = 1.0 + p7 * b
    
    # Interaction term
    inter_term = 1.0 + p8 * (temp - p3) * (pH - p5) / (sigma_T * sigma_pH + 1e-6)
    inter_term = np.clip(inter_term, 0.5, 2.0)
    
    # Combine
    result = s_poly * monod_gate * gauss_T * gauss_pH * dens_term * inter_term + p9
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
