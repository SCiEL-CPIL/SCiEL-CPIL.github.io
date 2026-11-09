"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a Generalized Additive Model (GAM) with 4 basis functions per input variable.
    The 10 parameters are:
    - params[0]: global bias
    - params[1:5]: coefficients for basis functions of b (population density)
    - params[5:9]: coefficients for basis functions of s (substrate)
    - params[9]:   coefficient for basis functions of temp (temperature) - wait, we only have 10 params total
    
    Let's redistribute: 
    - params[0]: global bias
    - params[1:4]: 3 basis coefficients for b
    - params[4:7]: 3 basis coefficients for s
    - params[7:9]: 2 basis coefficients for temp
    - params[9]:   1 basis coefficient for pH
    
    Actually, let's use 2 basis functions per variable to keep it simple and within 10 params:
    - params[0]: bias
    - params[1:3]: 2 bases for b
    - params[3:5]: 2 bases for s
    - params[5:7]: 2 bases for temp
    - params[7:9]: 2 bases for pH
    - params[9]:   interaction term b*s (or just another basis)
    
    Better yet, let's use 2 radial basis functions (RBFs) per variable, centered at fixed locations.
    The 10 parameters will be:
    - params[0]: bias
    - params[1], params[2]: coefficients for 2 RBFs on b
    - params[3], params[4]: coefficients for 2 RBFs on s
    - params[5], params[6]: coefficients for 2 RBFs on temp
    - params[7], params[8]: coefficients for 2 RBFs on pH
    - params[9]: coefficient for a mild interaction term (b * s) to capture synergy
    """
    # Define fixed centers for RBFs based on typical ranges
    # b: [1, 4.998] -> centers at 2.0, 4.0
    # s: [1, 4.999] to [1.002, 39.98] -> centers at 5.0, 20.0
    # temp: [1, 4.999] to [2, 12] -> centers at 5.0, 10.0
    # pH: [1, 4.998] to [1.002, 39.98] -> centers at 5.0, 20.0
    
    # RBF width (sigma) fixed to avoid overfitting and keep convexity
    sigma = 2.0
    
    # Basis functions for b
    b_center1 = 2.0
    b_center2 = 4.0
    rbf_b1 = np.exp(-((b - b_center1) ** 2) / (2 * sigma ** 2))
    rbf_b2 = np.exp(-((b - b_center2) ** 2) / (2 * sigma ** 2))
    
    # Basis functions for s
    s_center1 = 5.0
    s_center2 = 20.0
    rbf_s1 = np.exp(-((s - s_center1) ** 2) / (2 * sigma ** 2))
    rbf_s2 = np.exp(-((s - s_center2) ** 2) / (2 * sigma ** 2))
    
    # Basis functions for temp
    temp_center1 = 5.0
    temp_center2 = 10.0
    rbf_temp1 = np.exp(-((temp - temp_center1) ** 2) / (2 * sigma ** 2))
    rbf_temp2 = np.exp(-((temp - temp_center2) ** 2) / (2 * sigma ** 2))
    
    # Basis functions for pH
    ph_center1 = 5.0
    ph_center2 = 20.0
    rbf_ph1 = np.exp(-((pH - ph_center1) ** 2) / (2 * sigma ** 2))
    rbf_ph2 = np.exp(-((pH - ph_center2) ** 2) / (2 * sigma ** 2))
    
    # Interaction term (mild)
    interaction = b * s
    
    # Combine with parameters
    pred = (params[0] + 
            params[1] * rbf_b1 + params[2] * rbf_b2 +
            params[3] * rbf_s1 + params[4] * rbf_s2 +
            params[5] * rbf_temp1 + params[6] * rbf_temp2 +
            params[7] * rbf_ph1 + params[8] * rbf_ph2 +
            params[9] * interaction)
    
    return pred
# EVOLVE-BLOCK-END
