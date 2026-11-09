import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1]
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # s: [1, 5]
    xs = (s - 1.0) / 4.0
    # T: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # Unpack params
    A = p[0]
    K_s = p[1]
    T_opt = p[2]
    sigma_T = p[3]
    pH_opt = p[4]
    sigma_pH = p[5]
    c_b1 = p[6]
    c_b2 = p[7]
    c_b3 = p[8]
    offset = p[9]
    
    # Monod term for substrate s
    # Clip K_s to avoid division by zero or negative values
    K_s_safe = np.maximum(K_s, 1e-6)
    monod_s = xs / (K_s_safe + xs)
    
    # Gaussian term for temperature
    sigma_T_safe = np.maximum(sigma_T, 1e-6)
    gauss_T = np.exp(-((xT - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))
    
    # Gaussian term for pH
    sigma_pH_safe = np.maximum(sigma_pH, 1e-6)
    gauss_pH = np.exp(-((xPH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))
    
    # Polynomial term for biomass b
    poly_b = 1.0 + c_b1 * xb + c_b2 * xb**2
    
    # Combine
    y = A * monod_s * gauss_T * gauss_pH * poly_b + c_b3 * xb + offset
    
    return y
