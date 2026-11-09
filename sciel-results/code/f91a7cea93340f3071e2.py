import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid model: cubic polynomial + 2 Gaussian bumps
    # params: [p0, p1, p2, p3, mu1, log_sigma1, amp1, mu2, log_sigma2, amp2]
    # Actually let's use a cleaner parameterization:
    # p0, p1, p2, p3: polynomial coefficients
    # p4: mu1, p5: log_sigma1, p6: amp1
    # p7: mu2, p8: log_sigma2, p9: amp2
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    mu1 = params[4]
    log_sigma1 = params[5]
    amp1 = params[6]
    mu2 = params[7]
    log_sigma2 = params[8]
    amp2 = params[9]
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Gaussian 1
    # Clip log_sigma to keep sigma in a reasonable range [0.05, 10]
    log_sigma1_clipped = np.clip(log_sigma1, -3.0, 2.3)
    sigma1 = np.exp(log_sigma1_clipped)
    z1 = (x - mu1) / sigma1
    gauss1 = amp1 * np.exp(-0.5 * z1**2)
    
    # Gaussian 2
    log_sigma2_clipped = np.clip(log_sigma2, -3.0, 2.3)
    sigma2 = np.exp(log_sigma2_clipped)
    z2 = (x - mu2) / sigma2
    gauss2 = amp2 * np.exp(-0.5 * z2**2)
    
    return poly + gauss1 + gauss2
