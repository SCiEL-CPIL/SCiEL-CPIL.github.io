import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a multiplicative design with
    substrate (Monod), temperature (Gaussian), pH (Gaussian), and
    density (logistic inhibition) terms.
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: log(Ks) - substrate half-saturation
    p2: Temperature optimum
    p3: log(sigma_T) - temperature width
    p4: pH optimum
    p5: log(sigma_pH) - pH width
    p6: log(Kb) - density inhibition constant
    p7: logit(inhibition_strength)
    p8: Temperature-pH interaction strength
    p9: Bias term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth positive mappings
    Ks = np.exp(p1) + 1e-6
    sigma_T = np.exp(p3) + 1e-6
    sigma_pH = np.exp(p5) + 1e-6
    Kb = np.exp(p6) + 1e-6
    
    # Sigmoid for inhibition strength in (0, 1)
    inhib_strength = 1.0 / (1.0 + np.exp(-p7))
    
    # Substrate term: Monod
    S_term = s / (Ks + s)
    
    # Temperature term: Gaussian response (values in (0, 1])
    T_term = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH term: Gaussian response (values in (0, 1])
    pH_term = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Density inhibition term
    B_term = 1.0 - inhib_strength * (b / (Kb + b))
    
    # Interaction term: captures synergy/antagonism between T and pH
    # Use a smooth product-like interaction
    interaction = 1.0 + p8 * (T_term - 1.0) * (pH_term - 1.0)
    
    # Combine all terms
    growth = p0 * S_term * T_term * pH_term * B_term * interaction + p9
    
    return growth
