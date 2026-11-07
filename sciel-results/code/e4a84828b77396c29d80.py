import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Normalization
    T_n = (temp - 1.0) / 39.0
    pH_n = (pH - 2.0) / 10.0
    
    # Safe exponential parameters
    # Clip inputs to exp to prevent overflow/underflow issues
    # Underflow to 0 is fine, Overflow to Inf is bad.
    # So we mainly need to prevent large positive arguments.
    
    K_s = np.exp(np.clip(p[0], -50, 50))
    sigma_T = np.exp(np.clip(p[1], -50, 50))
    sigma_pH = np.exp(np.clip(p[2], -50, 50))
    mu_max = np.exp(np.clip(p[3], -50, 50))
    
    T_opt = p[4]
    pH_opt = p[5]
    dens_coeff = p[6]
    
    # Monod
    # Avoid division by zero if K_s is tiny? K_s >= exp(-50) ~ 1e-22.
    # s is >= 1. So K_s + s >= 1. Safe.
    monod = s / (K_s + s)
    
    # Temperature Gaussian
    # sigma_T >= exp(-50). If sigma is tiny, the term becomes a spike.
    # (T_n - T_opt)^2 is small.
    # If sigma is 0, division by zero.
    # But sigma >= 1e-22.
    # Let's ensure sigma is not too small to avoid numerical instability in the square?
    # Actually, if sigma is 1e-10, sigma^2 is 1e-20.
    # (T-T_opt)^2 / 2e-20. If T!=T_opt, this is huge. exp(-huge) = 0.
    # If T==T_opt, 0/0? No, 0 / 2e-20 = 0. exp(0)=1.
    # So it's a delta function.
    # This might be unstable for gradients.
    # Let's set a lower bound on sigma, e.g., exp(-5) ~ 0.006.
    sigma_T = np.maximum(sigma_T, 0.001)
    sigma_pH = np.maximum(sigma_pH, 0.001)
    
    temp_term = np.exp(-((T_n - T_opt) ** 2) / (2 * sigma_T ** 2))
    ph_term = np.exp(-((pH_n - pH_opt) ** 2) / (2 * sigma_pH ** 2))
    
    # Density
    # arg = -dens_coeff * b
    # b is 1-5.
    # If dens_coeff is -100, arg is 100-500. exp(500) is Inf.
    # So clip the argument.
    dens_arg = -dens_coeff * b
    dens_term = np.exp(np.clip(dens_arg, -50, 50))
    
    base = mu_max * monod * temp_term * ph_term * dens_term
    
    # Additive terms
    # p[7], p[8], p[9]
    # These can be any value.
    correction = p[7] * s + p[8] * temp + p[9]
    
    return base + correction
