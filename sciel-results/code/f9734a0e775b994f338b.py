# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear parametric model: sum of 3 Gaussians + linear term.
    Structurally different from linear basis approach.
    w1, w2, w3: Gaussian weights
    s1, s2, s3: log-sigmas (clamped for stability)
    a, b: linear coefficients
    m1, m2: Gaussian centers (third fixed at 0)
    """
    w1, w2, w3, s1, s2, s3, a, b, m1, m2 = params
    
    # Clamp log-sigmas to avoid numerical issues
    s1 = np.clip(s1, -10, 10)
    s2 = np.clip(s2, -10, 10)
    s3 = np.clip(s3, -10, 10)
    
    sigma1 = np.exp(s1)
    sigma2 = np.exp(s2)
    sigma3 = np.exp(s3)
    
    g1 = w1 * np.exp(-((x - m1)**2) / (2.0 * sigma1**2))
    g2 = w2 * np.exp(-((x - m2)**2) / (2.0 * sigma2**2))
    g3 = w3 * np.exp(-(x**2) / (2.0 * sigma3**2))
    
    return g1 + g2 + g3 + a * x + b
# EVOLVE-BLOCK-END
