# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Quadratic polynomial model for E. coli growth rate.
    
    Uses normalized inputs centered on the training data ranges.
    The model includes:
    - Constant term
    - Linear terms for all 4 inputs
    - Squared terms for temperature and pH (capturing optima)
    - Interaction terms for (b,s), (s,temp), (b,temp)
    
    Total parameters: 10
    """
    # Normalize inputs based on provided training ranges
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
    
    # Compute basis functions
    # 1. Constant
    const = np.ones_like(b_norm)
    
    # 2-5. Linear
    lin_b = b_norm
    lin_s = s_norm
    lin_t = temp_norm
    lin_p = pH_norm
    
    # 6-7. Quadratic (Optima)
    quad_t = temp_norm ** 2
    quad_p = pH_norm ** 2
    
    # 8-10. Interactions
    int_bs = b_norm * s_norm
    int_st = s_norm * temp_norm
    int_bt = b_norm * temp_norm
    
    # Combine
    y = (p0 * const + 
         p1 * lin_b + 
         p2 * lin_s + 
         p3 * lin_t + 
         p4 * lin_p + 
         p5 * quad_t + 
         p6 * quad_p + 
         p7 * int_bs + 
         p8 * int_st + 
         p9 * int_bt)
         
    return y
# EVOLVE-BLOCK-END
