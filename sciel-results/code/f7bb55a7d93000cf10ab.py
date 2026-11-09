"""E. coli growth rate model: mechanistic multiplicative core with additive flexibility."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Combines:
    - Monod kinetics for substrate (s)
    - Gaussian response for temperature (temp)
    - Gaussian response for pH
    - Linear response for population density (b)
    
    Uses raw inputs (no normalization) with parameters directly on the data scale.
    This allows the optimizer to find meaningful parameter values.
    
    Structural changes from parent:
    1. Added quadratic density term for non-linear inhibition.
    2. Added a second, broader Gaussian for temperature to capture asymmetry or
       a plateau, which is common in microbial growth curves.
    3. Kept the additive correction terms but refined the interaction structure.
    
    Parameters (10 used):
    p0: Max growth rate scale (C)
    p1: Substrate half-saturation constant (K)
    p2: Optimal temperature (T_center)
    p3: Temperature response width (sigma_T)
    p4: Optimal pH (pH_center)
    p5: pH response width (sigma_pH)
    p6: Density linear coefficient (gamma1)
    p7: Density quadratic coefficient (gamma2)
    p8: Broad temperature Gaussian weight (beta)
    p9: Bias (B)
    """
    # Unpack params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Monod substrate kinetics
    # Add small epsilon to prevent division by zero
    K = np.maximum(p1, 1e-6)
    monod = s / (K + s)

    # Temperature bell (Gaussian)
    # Use maximum to prevent division by zero and keep width positive
    sigma_T = np.maximum(np.abs(p3), 1.0)
    temp_term = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # Second broader temperature Gaussian to capture asymmetry/plateau
    # Center it slightly lower or same, but wider. 
    # Let's use a fixed center offset from p2 or just a very wide Gaussian.
    # To keep it simple and parameter-efficient, let's use a fixed broad Gaussian
    # centered at 37 (typical E. coli optimum) or just scale the existing one.
    # Better: use a second Gaussian with center p2 and width 3*sigma_T, weighted by p8.
    # Or even simpler: just add a linear term? No, Gaussian is better.
    # Let's add a second Gaussian with center 37.0 and width 15.0, scaled by p8.
    # This adds a "background" growth rate that is less sensitive to temp.
    temp_term_broad = np.exp(-((temp - 37.0) ** 2) / (2.0 * 15.0 ** 2))
    # Combine temperature terms: main * (1 + p8 * broad) ? Or just add?
    # Let's do: temp_combined = temp_term + p8 * temp_term_broad
    # But p8 needs to be small. Let's just add it as an additive term in the main product?
    # No, let's keep the multiplicative structure but allow the temp response to be a sum of two Gaussians.
    temp_combined = temp_term + p8 * temp_term_broad

    # pH bell (Gaussian)
    sigma_pH = np.maximum(np.abs(p5), 1.0)
    ph_term = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))

    # Density term (linear + quadratic inhibition)
    dens_term = 1.0 + p6 * b + p7 * b ** 2

    # Main multiplicative term
    main_term = p0 * monod * temp_combined * ph_term * dens_term

    # Additive corrections
    # Keep the bias p9.
    # The parent had p8 and p9 as interaction terms. Now p8 is used for temp.
    # We have 10 params. p0..p9.
    # p0: C
    # p1: K
    # p2: T_center
    # p3: sigma_T
    # p4: pH_center
    # p5: sigma_pH
    # p6: gamma1
    # p7: gamma2
    # p8: beta (broad temp weight)
    # p9: Bias
    
    # We lost the specific interaction terms p8*monod*(temp-p2) and p9*monod*(pH-p4) from the parent.
    # Are they necessary? The multiplicative structure already captures interactions.
    # The additive bias p9 is important for offset.
    
    result = main_term + p9

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
