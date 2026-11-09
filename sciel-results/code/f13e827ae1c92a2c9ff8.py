"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH."""
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe parameters to avoid division by zero or negative values in exp/log
    # p1 (Ks) should be positive
    Ks = np.maximum(p1, 1e-6)
    
    # p3 (sigma_T) should be positive
    sigma_T = np.maximum(np.abs(p3), 1e-6)
    
    # p5 (sigma_pH) should be positive
    sigma_pH = np.maximum(np.abs(p5), 1e-6)
    
    # p6 (K_b) should be positive
    K_b = np.maximum(p6, 1e-6)
    
    # p7 (Hill coefficient) should be positive to ensure monotonic decay in inhibition term
    # If p7 is negative, the term flips. We want inhibition to increase with b.
    # Hill term: K^n / (K^n + b^n). If n > 0, as b -> inf, term -> 0.
    # If n < 0, as b -> inf, term -> inf (if we don't clip).
    # Let's force n to be positive for inhibition.
    n_hill = np.maximum(p7, 1e-3)
    
    # p9 (Ki) for substrate inhibition.
    # Term: s / (Ks + s + s^2/Ki)
    # If Ki -> inf, term -> s / (Ks + s) (Monod)
    # If Ki is small, s^2/Ki dominates at high s, causing inhibition.
    # We need Ki to be positive to make sense in this denominator.
    # If p9 is negative, the denominator could become zero or negative.
    # Let's use abs(p9) + epsilon to ensure it's positive.
    Ki = np.maximum(np.abs(p9), 1e-6)
    
    # 1. Substrate Response (Haldane Kinetics)
    # Bounded, smooth, allows for inhibition at high s
    # s is in [1, 5], so s^2 is small.
    substrate_term = s / (Ks + s + (s ** 2) / Ki)
    
    # 2. Temperature Response (Gaussian)
    temp_term = np.exp(-((temp - p2) ** 2) / (2.0 * sigma_T ** 2))
    
    # 3. pH Response (Gaussian)
    ph_term = np.exp(-((pH - p4) ** 2) / (2.0 * sigma_pH ** 2))
    
    # 4. Density Response (Hill Function Inhibition)
    # Standard form: K^n / (K^n + b^n)
    # This is 1 when b=0 and approaches 0 as b increases.
    # Using powers can be unstable if b is large and n is large, but b is [1, 5] and n is likely small (1-3).
    # To be safe against overflow in b**n, we can rewrite:
    # 1 / (1 + (b/K)^n)
    density_term = 1.0 / (1.0 + (b / K_b) ** n_hill)
    
    # Combine
    growth = p0 * substrate_term * temp_term * ph_term * density_term
    
    # Add offset
    growth += p8
    
    # Ensure finite values
    growth = np.where(np.isfinite(growth), growth, 0.0)
    
    return growth
# EVOLVE-BLOCK-END
