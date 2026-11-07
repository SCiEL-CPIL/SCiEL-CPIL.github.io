import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics for substrate,
    Gaussian responses for temperature and pH with skewness,
    and exponential inhibition for population density.
    
    Parameters are centered to facilitate optimization from initial guess of 1s.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Centered Optima based on known input ranges
    # T range [1, 40], center ~20.5
    T_opt = 20.0 + p2
    # pH range [2, 12], center ~7
    pH_opt = 7.0 + p4
    
    # Positive parameters via exponential
    K_s = np.exp(p1)
    sigma_T = np.exp(p3)
    sigma_pH = np.exp(p5)
    k_b = np.exp(p6)
    
    # Substrate (Monod)
    # s is in [1, 5]
    monod_s = s / (K_s + s)
    
    # Temperature Response (Skewed Gaussian)
    z_T = (temp - T_opt) / sigma_T
    gauss_T = np.exp(-0.5 * z_T**2 - p8 * z_T)
    
    # pH Response (Skewed Gaussian)
    z_pH = (pH - pH_opt) / sigma_pH
    gauss_pH = np.exp(-0.5 * z_pH**2 - p9 * z_pH)
    
    # Density Inhibition
    # b is in [1, 5]
    inhib_b = np.exp(-k_b * b)
    
    # Combine
    pred = p0 * monod_s * gauss_T * gauss_pH * inhib_b + p7
    
    return pred
