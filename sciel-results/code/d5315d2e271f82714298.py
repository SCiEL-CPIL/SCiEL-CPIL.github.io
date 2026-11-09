import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Generalized Additive Model (GAM) with RBF bases.
    
    This approach uses a linear combination of Radial Basis Functions (Gaussian kernels)
    for each input variable. The 10 parameters are the coefficients for these bases.
    
    Model:
    y = sum_j w_j * exp(-((x_j - c_j)^2) / (2 * sigma_j^2))
    
    where x_j are the normalized inputs, c_j are centers, sigma_j are widths,
    and w_j are the learnable weights (params).
    
    We use 3 basis functions for temperature (most critical) and 1 for each other variable,
    plus an intercept, totaling 10 parameters:
    p[0]: intercept
    p[1]: weight for b at center 0
    p[2]: weight for s at center 0
    p[3]: weight for T at center -0.8 (cold)
    p[4]: weight for T at center 0.2 (optimal ~30C)
    p[5]: weight for T at center 0.9 (hot)
    p[6]: weight for pH at center 0
    p[7]: weight for b at center 0.5
    p[8]: weight for s at center 0.5
    p[9]: weight for T at center -0.3
    
    This additive structure is linear in parameters, making the optimization problem
    convex and easy for BFGS to solve.
    """
    p = params
    
    # Normalize inputs to roughly [-1, 1]
    b_n = (b - 3.0) / 2.0
    s_n = (s - 3.0) / 2.0
    T_n = (temp - 20.5) / 19.5
    pH_n = (pH - 7.0) / 5.0
    
    # Define RBF centers and widths
    # Centers chosen to capture different regimes
    # Widths chosen to be broad enough to be smooth but specific enough to capture trends
    
    # RBF function: exp(-(x - center)^2 / (2 * width^2))
    
    # For b: 2 basis functions
    b_rbf1 = np.exp(-((b_n - 0.0) ** 2) / (2 * 0.8 ** 2))
    b_rbf2 = np.exp(-((b_n - 0.5) ** 2) / (2 * 0.8 ** 2))
    
    # For s: 2 basis functions
    s_rbf1 = np.exp(-((s_n - 0.0) ** 2) / (2 * 0.8 ** 2))
    s_rbf2 = np.exp(-((s_n - 0.5) ** 2) / (2 * 0.8 ** 2))
    
    # For T: 4 basis functions (temperature is critical)
    T_rbf1 = np.exp(-((T_n - (-0.8)) ** 2) / (2 * 0.5 ** 2))  # Cold
    T_rbf2 = np.exp(-((T_n - (-0.3)) ** 2) / (2 * 0.5 ** 2))  # Mild
    T_rbf3 = np.exp(-((T_n - 0.2) ** 2) / (2 * 0.5 ** 2))     # Optimal
    T_rbf4 = np.exp(-((T_n - 0.9) ** 2) / (2 * 0.5 ** 2))     # Hot
    
    # For pH: 1 basis function
    pH_rbf1 = np.exp(-((pH_n - 0.0) ** 2) / (2 * 1.0 ** 2))
    
    # Linear combination
    mu = (p[0] 
          + p[1] * b_rbf1 
          + p[2] * s_rbf1 
          + p[3] * T_rbf1 
          + p[4] * T_rbf3 
          + p[5] * T_rbf4 
          + p[6] * pH_rbf1 
          + p[7] * b_rbf2 
          + p[8] * s_rbf2 
          + p[9] * T_rbf2)
          
    return mu
