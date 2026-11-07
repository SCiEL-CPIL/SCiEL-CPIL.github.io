import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining soft-min limiting-factor dynamics
    with a bilinear T-pH interaction term for synergy/antagonism capture.
    
    Parameters (10 total):
    p0: Max growth rate scale
    p1: log(Ks) - substrate half-saturation
    p2: Temperature optimum
    p3: log(sigma_T) - temperature width
    p4: pH optimum
    p5: log(sigma_pH) - pH width
    p6: log(Kb) - density inhibition constant
    p7: logit(inhibition_strength)
    p8: Bilinear T-pH interaction coefficient
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
    
    # Bilinear interaction: captures synergy/antagonism between T and pH
    # When both are suboptimal (T_term, pH_term < 1), the product of deviations
    # is positive, and p8 scales this correction
    interaction = 1.0 + p8 * (T_term - 1.0) * (pH_term - 1.0)
    
    # Soft-min of T and pH terms (limiting factor dynamics)
    # Use log-sum-exp for numerical stability
    # soft_min(T, P) = -1/k * log((exp(-k*T) + exp(-k*P)) / 2)
    # We fix k to a reasonable value for stability, or use a moderate k
    # Since we removed the k parameter (p8 is now interaction), we use a fixed k
    # Actually, let's reconsider: we have 10 params. Let's keep the soft-min
    # but use a fixed sharpness, or better, combine the interaction with the product.
    
    # Alternative: Use the product T*pH with the interaction term, 
    # which is smoother and easier for BFGS than soft-min with large k.
    # The interaction term 1 + p8*(T-1)*(pH-1) allows the model to adjust
    # the degree to which T and pH are independent or coupled.
    
    # Let's stick with the multiplicative form with interaction, as it's smoother
    growth = p0 * S_term * T_term * pH_term * B_term * interaction + p9
    
    return growth
