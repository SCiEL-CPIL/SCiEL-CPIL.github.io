import numpy as np

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate model using a product of univariate smooth functions.
    
    Structure: mu = mu_max * f_s(s) * f_t(temp) * f_pH(pH) * f_b(b)
    
    Each f_i is a smooth univariate function built from Gaussian basis functions.
    Parameters:
        p[0]: mu_max (maximum growth rate)
        p[1]: K_s (Monod half-saturation for substrate)
        p[2]: T_opt (optimal temperature)
        p[3]: sigma_T (temperature tolerance width)
        p[4]: pH_opt (optimal pH)
        p[5]: sigma_pH (pH tolerance width)
        p[6]: K_b (density inhibition constant)
        p[7]: exponent for substrate saturation
        p[8]: exponent for temperature/pH Gaussian
        p[9]: offset
    """
    p = params
    
    # Ensure positive values for constants that need them
    mu_max = p[0]
    K_s = np.abs(p[1]) + 1e-8
    T_opt = p[2]
    sigma_T = np.abs(p[3]) + 1e-8
    pH_opt = p[4]
    sigma_pH = np.abs(p[5]) + 1e-8
    K_b = np.abs(p[6]) + 1e-8
    exp_s = np.abs(p[7]) + 0.1
    exp_t = np.abs(p[8]) + 0.1
    offset = p[9]
    
    # Substrate: Monod-like saturation
    # f_s(s) = s^exp_s / (K_s^exp_s + s^exp_s)
    s_safe = np.maximum(s, 1e-8)
    f_s = (s_safe ** exp_s) / (K_s ** exp_s + s_safe ** exp_s)
    
    # Temperature: Gaussian centered at T_opt
    # f_t(temp) = exp(-0.5 * ((temp - T_opt) / sigma_T)^2 * exp_t)
    t_diff = (temp - T_opt) / sigma_T
    f_t = np.exp(-0.5 * t_diff * t_diff * exp_t)
    
    # pH: Gaussian centered at pH_opt
    # f_pH(pH) = exp(-0.5 * ((pH - pH_opt) / sigma_pH)^2 * exp_t)
    ph_diff = (pH - pH_opt) / sigma_pH
    f_ph = np.exp(-0.5 * ph_diff * ph_diff * exp_t)
    
    # Density: Inhibition at high density
    # f_b(b) = 1 / (1 + (b / K_b)^exp_s)
    b_safe = np.maximum(b, 1e-8)
    f_b = 1.0 / (1.0 + (b_safe / K_b) ** exp_s)
    
    # Combine all factors
    mu = mu_max * f_s * f_t * f_ph * f_b + offset
    
    # Ensure non-negative growth rate
    mu = np.maximum(mu, 0.0)
    
    return mu
