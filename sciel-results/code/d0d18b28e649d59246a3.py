"""Symbolic regression with a hybrid basis of scaled polynomials, sinusoids, and Gaussians."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Scale x to [-1, 1] for better conditioned polynomial terms
    u = x / 2.0
    
    # Global trend: Scaled polynomials (up to degree 4)
    # Using u keeps values bounded in [-1, 1]
    term0 = np.ones_like(x)
    term1 = u
    term2 = u ** 2
    term3 = u ** 3
    term4 = u ** 4
    
    # Periodic component
    term5 = np.sin(x)
    
    # Local components: 4 Gaussians centered at -1.5, -0.5, 0.5, 1.5
    # Width sigma^2 = 0.5
    sigma2 = 0.5
    term6 = np.exp(-((x - (-1.5)) ** 2) / sigma2)
    term7 = np.exp(-((x - (-0.5)) ** 2) / sigma2)
    term8 = np.exp(-((x - (0.5)) ** 2) / sigma2)
    term9 = np.exp(-((x - (1.5)) ** 2) / sigma2)
    
    # Combine
    result = (p[0] * term0 + 
              p[1] * term1 + 
              p[2] * term2 + 
              p[3] * term3 + 
              p[4] * term4 + 
              p[5] * term5 + 
              p[6] * term6 + 
              p[7] * term7 + 
              p[8] * term8 + 
              p[9] * term9)
              
    return result
# EVOLVE-BLOCK-END
