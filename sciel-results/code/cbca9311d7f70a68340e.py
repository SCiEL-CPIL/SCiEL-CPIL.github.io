import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod substrate, Gaussian T/pH,
    density inhibition, and a soft-min interaction between T and pH
    to capture limiting factor dynamics.
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: log(Ks) - substrate half-saturation
    p2: Temperature optimum
    p3: log(sigma_T) - temperature width
    p4: pH optimum
    p5: log(sigma_pH) - pH width
    p6: log(Kb) - density inhibition constant
    p7: logit(inhibition_strength)
    p8: log(k) - soft-min sharpness
    p9: Bias term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Smooth positive mappings
    Ks = np.exp(p1) + 1e-6
    sigma_T = np.exp(p3) + 1e-6
    sigma_pH = np.exp(p5) + 1e-6
    Kb = np.exp(p6) + 1e-6
    k = np.exp(p8) + 1e-6  # Sharpness of soft-min
    
    # Sigmoid for inhibition strength in (0, 1)
    inhib_strength = 1.0 / (1.0 + np.exp(-p7))
    
    # Substrate term: Monod
    S_term = s / (Ks + s)
    
    # Temperature term: Gaussian response (values in (0, 1])
    T_term = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # pH term: Gaussian response (values in (0, 1])
    pH_term = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))
    
    # Soft-min interaction: captures that the worst condition limits growth
    # soft_min(T, P) = -1/k * log( (exp(-k*T) + exp(-k*P)) / 2 )
    # This is smooth and approximates min(T,P) as k -> inf
    # Note: T_term and pH_term are <= 1, so exp(-k*T) is >= exp(-k) > 0
    # To avoid overflow in exp for large k, we can use log-sum-exp trick
    # log(exp(-kT) + exp(-kP)) = -k*max(T,P) + log(exp(-k(T-max)) + exp(-k(P-max)))
    # However, T, P are small (<=1), k might be large.
    # Let's just use np.logaddexp for stability
    # log(exp(-k*T) + exp(-k*P)) = logaddexp(-k*T, -k*P)
    # soft_min = -1/k * (logaddexp(-k*T, -k*P) - log(2))
    
    log_sum = np.logaddexp(-k * T_term, -k * pH_term)
    soft_min_TP = -(log_sum - np.log(2.0)) / k
    
    # Density inhibition term
    B_term = 1.0 - inhib_strength * (b / (Kb + b))
    
    # Combine all terms
    growth = p0 * S_term * soft_min_TP * B_term + p9
    
    return growth
