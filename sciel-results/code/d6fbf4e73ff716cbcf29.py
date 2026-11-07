"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod substrate kinetics,
    Gaussian temperature and pH optima, and density-dependent inhibition.
    
    The model is:
        mu = p0 * (s / (Ks + s)) * exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
             * exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
             * (1 - (b / Kb)^n)
             + p7 * b + p8 * s + p9
    
    Parameters:
        b: Population density
        s: Substrate concentration
        temp: Temperature
        pH: pH level
        params: Array of 10 fitted constants.
            p[0]: Max growth rate scale
            p[1]: Log of Monod half-saturation constant (Ks)
            p[2]: Optimal temperature
            p[3]: Log of temperature width (sigma_T)
            p[4]: Optimal pH
            p[5]: Log of pH width (sigma_pH)
            p[6]: Log of density inhibition constant (Kb)
            p[7]: Linear coefficient for density
            p[8]: Linear coefficient for substrate
            p[9]: Baseline offset
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Monod term: s / (Ks + s), with Ks = exp(p1)
    Ks = np.exp(p1)
    monod = s / (Ks + s)
    
    # Temperature Gaussian: exp(-((temp - T_opt)^2) / (2 * sigma_T^2))
    # sigma_T = exp(p3)
    sigma_T = np.exp(p3)
    temp_term = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH Gaussian: exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
    # sigma_pH = exp(p5)
    sigma_pH = np.exp(p5)
    ph_term = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition: (1 - (b / Kb)^n)
    # Kb = exp(p6), n is fixed at 1 for simplicity (could use another param but we have 10)
    Kb = np.exp(p6)
    density_term = 1.0 - (b / Kb)
    
    # Combine all terms
    # Ensure non-negative growth rate contribution from the main term
    main_term = p0 * monod * temp_term * ph_term * density_term
    
    # Add linear corrections for better fit
    linear_term = p7 * b + p8 * s + p9
    
    return main_term + linear_term
# EVOLVE-BLOCK-END
