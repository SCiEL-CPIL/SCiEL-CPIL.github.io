import numpy as np

# SEARCH-START
def equation(b, s, temp, pH, params):
    """
    E. coli growth rate prediction using a linear combination of fixed
    biological basis functions (Monod and RBF).
    
    This model is linear in the 10 parameters, ensuring a convex loss
    landscape for BFGS optimization.
    
    Basis Functions:
    1. Intercept
    2. Monod saturation for Substrate (S)
    3. Monod saturation for Biomass (B)
    4-6. Gaussian RBFs for Temperature (centers: 20, 30, 37)
    7-9. Gaussian RBFs for pH (centers: 5, 7, 9)
    10. Interaction between optimal Temp and optimal pH
    """
    p = params
    
    # Ensure inputs are float arrays for safe operations
    b = np.asarray(b, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    pH = np.asarray(pH, dtype=np.float64)
    
    # 1. Intercept
    f1 = np.ones_like(b)
    
    # 2. Monod for Substrate: S / (S + Ks), fixed Ks=1.0
    # Clipping S to avoid negative values if any (though range is 1-5)
    s_safe = np.maximum(s, 1e-6)
    f2 = s_safe / (s_safe + 1.0)
    
    # 3. Monod for Biomass: B / (B + Kb), fixed Kb=1.0
    # Assumption: Growth increases with B up to a point.
    b_safe = np.maximum(b, 1e-6)
    f3 = b_safe / (b_safe + 1.0)
    
    # 4-6. RBFs for Temperature
    # Centers chosen to cover the physiological range and optimum
    # Width = 5.0
    sigma_t = 5.0
    f4 = np.exp(-((temp - 20.0) ** 2) / (2.0 * sigma_t ** 2))
    f5 = np.exp(-((temp - 30.0) ** 2) / (2.0 * sigma_t ** 2))
    f6 = np.exp(-((temp - 37.0) ** 2) / (2.0 * sigma_t ** 2))
    
    # 7-9. RBFs for pH
    # Centers chosen to cover acidic, neutral, and alkaline conditions
    # Width = 1.5
    sigma_p = 1.5
    f7 = np.exp(-((pH - 5.0) ** 2) / (2.0 * sigma_p ** 2))
    f8 = np.exp(-((pH - 7.0) ** 2) / (2.0 * sigma_p ** 2))
    f9 = np.exp(-((pH - 9.0) ** 2) / (2.0 * sigma_p ** 2))
    
    # 10. Interaction Term: Coupling the optimal temperature and pH responses
    # This allows the model to capture if the pH optimum shifts with temperature
    f10 = f6 * f8
    
    # Linear combination of basis functions
    pred = (p[0] * f1 + 
            p[1] * f2 + 
            p[2] * f3 + 
            p[3] * f4 + 
            p[4] * f5 + 
            p[5] * f6 + 
            p[6] * f7 + 
            p[7] * f8 + 
            p[8] * f9 + 
            p[9] * f10)
            
    return pred
# SEARCH-END
