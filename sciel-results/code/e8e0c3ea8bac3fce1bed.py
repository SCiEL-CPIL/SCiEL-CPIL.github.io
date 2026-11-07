# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Enhanced model for E. coli growth rate using a combination of:
    - Normalized inputs with better scaling
    - Gaussian-like peaked functions for temperature and pH optima
    - Logarithmic terms for nutrient concentrations (b, s)
    - Interaction terms
    - Sigmoid-like saturation for nutrients
    
    Uses 10 parameters total.
    """
    # Deadline guard
    t_start = time.time()
    
    # Normalize inputs
    # b: [1, 4.998] -> center 3.0, scale 2.0
    # s: [1, 4.999] -> center 3.0, scale 2.0
    # temp: [1.002, 39.98] -> center 20.5, scale 10.0
    # pH: [2, 12] -> center 7.0, scale 5.0
    
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 3.0) / 2.0
    temp_norm = (temp - 20.5) / 10.0
    pH_norm = (pH - 7.0) / 5.0
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basis functions:
    # 1. Constant
    const = np.ones_like(b_norm)
    
    # 2. Log-saturation for b (nutrient)
    # Using tanh to model saturation
    sat_b = np.tanh(b_norm)
    
    # 3. Log-saturation for s (nutrient)
    sat_s = np.tanh(s_norm)
    
    # 4. Gaussian peak for temperature (optimum around center)
    # exp(-alpha * x^2) where alpha is fitted via p4
    gauss_t = np.exp(-np.abs(p4) * (temp_norm ** 2))
    
    # 5. Gaussian peak for pH (optimum around center)
    # exp(-beta * x^2) where beta is fitted via p5
    gauss_p = np.exp(-np.abs(p5) * (pH_norm ** 2))
    
    # 6. Linear temp (captures asymmetry if needed)
    lin_t = temp_norm
    
    # 7. Linear pH (captures asymmetry if needed)
    lin_p = pH_norm
    
    # 8. Interaction: sat_b * sat_s (nutrient co-limitation)
    int_bs = sat_b * sat_s
    
    # 9. Interaction: sat_b * gauss_t (nutrient-temperature coupling)
    int_bt = sat_b * gauss_t
    
    # 10. Interaction: sat_s * gauss_p (nutrient-pH coupling)
    int_sp = sat_s * gauss_p
    
    # Combine
    y = (p0 * const + 
         p1 * sat_b + 
         p2 * sat_s + 
         p3 * gauss_t + 
         p4 * lin_t + 
         p5 * gauss_p + 
         p6 * lin_p + 
         p7 * int_bs + 
         p8 * int_bt + 
         p9 * int_sp)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 145.0:
        # Return zeros as fallback (shouldn't happen for simple computation)
        return np.zeros_like(b)
    
    return y
# EVOLVE-BLOCK-END
