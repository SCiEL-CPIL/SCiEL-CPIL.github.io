import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid model: Chebyshev polynomial (deg 3) + 2 adaptive Gaussian bumps
    # params: [c0, c1, c2, c3, mu1, log_sigma1, amp1, mu2, log_sigma2, amp2]
    
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    
    mu1 = params[4]
    log_sigma1 = params[5]
    amp1 = params[6]
    
    mu2 = params[7]
    log_sigma2 = params[8]
    amp2 = params[9]
    
    # Scale x to [-1, 1] for Chebyshev stability
    # x is in [-3, 3], so t = x / 3 is in [-1, 1]
    t = x / 3.0
    
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    
    poly = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3
    
    # Gaussian 1
    # Clip log_sigma to keep sigma in a reasonable range
    # Original x range is [-3, 3]. If sigma is in t-space, it needs scaling back?
    # No, the Gaussians are in x-space directly in the current code.
    # It's better to keep Gaussians in x-space to allow wide/narrow peaks in the original domain.
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
