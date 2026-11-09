import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics for substrate,
    Gaussian bumps for temperature and pH, and a saturating term for density.
    
    Parameters:
    p0: Max growth rate scale
    p1: Ks (Monod constant for substrate)
    p2: Optimal Temperature
    p3: Log-sigma for Temperature (sigma_T = exp(p3))
    p4: Optimal pH
    p5: Log-sigma for pH (sigma_pH = exp(p5))
    p6: Offset
    p7: Unused/Interaction placeholder (set to 0 in init, might fit small noise)
    p8: Kb (Saturation constant for density)
    p9: Interaction term coefficient (T * pH)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure positive constants for denominators and sigmas
    # p1 (Ks) should be > 0. Use exp? Or just clamp?
    # BFGS doesn't constrain. If p1 becomes negative, division issues.
    # Let's use exp for Ks and Kb to ensure positivity.
    # But if we use exp, the scale changes.
    # Let's try: Ks = exp(p1), Kb = exp(p8)
    # Sigmas: sigma_T = exp(p3), sigma_pH = exp(p5)
    
    Ks = np.exp(p1)
    Kb = np.exp(p8)
    sigma_T = np.exp(p3)
    sigma_pH = np.exp(p5)
    
    # Monod term for substrate
    # s is in [1, 5]
    monod_s = s / (Ks + s)
    
    # Gaussian for Temperature
    # temp is in [1, 40]
    # p2 is T_opt
    temp_term = np.exp(-0.5 * ((temp - p2) / sigma_T) ** 2)
    
    # Gaussian for pH
    # pH is in [2, 12]
    # p4 is pH_opt
    ph_term = np.exp(-0.5 * ((pH - p4) / sigma_pH) ** 2)
    
    # Saturating term for density
    # b is in [1, 5]
    # If b is small, growth might be limited? Or if b is high, inhibition?
    # Standard Monod-like for density: b / (Kb + b)
    # This goes from 0 to 1.
    # If we want it to be 1 at low density, we could use 1 / (1 + b/Kb) which is inhibition.
    # Let's stick to b / (Kb + b) as a "biomass factor" if it's total rate, 
    # or just a scaling factor.
    # Given the seed was linear, and 1.0 score, let's try a simple linear + constant 
    # or the saturating form.
    # Let's use: 1 + p7 * (b / (Kb + b)) ? No, p7 is unused.
    # Let's use: (b / (Kb + b))
    
    density_term = b / (Kb + b)
    
    # Interaction term
    interaction = p9 * temp * pH
    
    # Combine
    # p0 scales the whole thing
    # p6 is offset
    # p7 is currently unused in this formulation, let's add it as a linear temp correction?
    # Or just leave it as a multiplier?
    # Let's add p7 as a linear term for temp to allow asymmetry if Gaussian is not enough?
    # Or just p7 * s?
    # Let's make p7 a linear term for density to complement the saturating one?
    # No, let's keep it simple.
    
    # Revised combination:
    # y = p0 * monod_s * temp_term * ph_term * density_term + p6 + p7 * b + p9 * temp * pH
    
    y = p0 * monod_s * temp_term * ph_term * density_term + p6 + p7 * b + p9 * temp * pH
    
    # Ensure finite
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
