import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a flexible additive-multiplicative hybrid:
    - Substrate: Monod
    - Temperature: Asymmetric Gaussian (left/right widths)
    - pH: Asymmetric Gaussian (left/right widths)
    - Density: Haldane-type inhibition (b/(Kb+b)) with optional quadratic term
    - Interaction: bilinear T*pH cross term
    - Bias
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: log(Ks) - substrate half-saturation
    p2: Temperature optimum
    p3: log(sigma_T_left) - temperature left width
    p4: log(sigma_T_right) - temperature right width
    p5: pH optimum
    p6: log(sigma_pH) - pH width (symmetric for simplicity)
    p7: log(Kb) - density inhibition constant
    p8: T-pH interaction coefficient
    p9: Bias term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Positive mappings
    Ks = np.exp(p1) + 1e-6
    sigma_T_left = np.exp(p3) + 1e-6
    sigma_T_right = np.exp(p4) + 1e-6
    sigma_pH = np.exp(p6) + 1e-6
    Kb = np.exp(p7) + 1e-6
    
    # Substrate term: Monod
    S_term = s / (Ks + s)
    
    # Asymmetric temperature Gaussian
    # For temp < p2: use sigma_T_left
    # For temp >= p2: use sigma_T_right
    diff_T = temp - p2
    sigma_T = np.where(diff_T < 0, sigma_T_left, sigma_T_right)
    T_term = np.exp(-0.5 * (diff_T / sigma_T) ** 2)
    
    # Symmetric pH Gaussian
    pH_term = np.exp(-0.5 * ((pH - p5) / sigma_pH) ** 2)
    
    # Density inhibition: Haldane-like
    B_term = 1.0 / (1.0 + (b / Kb))
    
    # Bilinear interaction between T and pH deviations
    interaction = 1.0 + p8 * (T_term - 1.0) * (pH_term - 1.0)
    
    # Combine
    growth = p0 * S_term * T_term * pH_term * B_term * interaction + p9
    
    return growth
