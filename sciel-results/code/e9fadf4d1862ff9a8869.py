import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid model: Polynomial (degree 2) + 2 Adaptive Gaussian Bumps
    # Parameter allocation (10 params):
    # p0, p1, p2: Polynomial coefficients
    # mu1, log_sigma1, amp1: Gaussian 1
    # mu2, log_sigma2, amp2: Gaussian 2
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    
    mu1 = params[3]
    log_sigma1 = params[4]
    amp1 = params[5]
    
    mu2 = params[6]
    log_sigma2 = params[7]
    amp2 = params[8]
    
    # Parameter 9 is unused in this simplified allocation, or we can use it for something else?
    # Wait, the previous code used 10 params. Let's stick to a robust 10-param usage.
    # Let's use a cubic polynomial to utilize the 10th parameter?
    # Or maybe 3 Gaussians? No, 2 is safer for BFGS.
    # Let's try: p0, p1, p2, p3 (cubic) + mu1, log_sig1, amp1 + mu2, log_sig2, amp2 = 10 params.
    # This matches the previous logic but cleaner.
    
    p3 = params[9]
    
    # Polynomial part
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3
    
    # Gaussian 1
    # Clip log_sigma to keep sigma in [0.1, 10]
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
