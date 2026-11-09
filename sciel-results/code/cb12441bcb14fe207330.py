"""Symbolic regression: polynomial + RBF + sinusoidal basis with robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model: polynomial (degree 0-4) + 3 Gaussian RBFs + sin/cos terms.
    
    10 parameters total:
      p[0..4]: polynomial coefficients (1, x, x^2, x^3, x^4)
      p[5..7]: Gaussian RBF weights at centers -1, 0, 1
      p[8]:   sin(x) weight
      p[9]:   cos(x) weight
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has at least 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Polynomial terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Gaussian RBF terms with moderate width
    sigma2 = 2.0 * 0.7**2  # 2 * sigma^2 for convenience
    rbf0 = np.exp(-(x + 1.0)**2 / sigma2)
    rbf1 = np.exp(-x2 / sigma2)
    rbf2 = np.exp(-(x - 1.0)**2 / sigma2)
    
    # Sinusoidal terms
    sx = np.sin(x)
    cx = np.cos(x)
    
    # Combine all basis functions
    y = (p[0] + 
         p[1] * x + 
         p[2] * x2 + 
         p[3] * x3 + 
         p[4] * x4 +
         p[5] * rbf0 + 
         p[6] * rbf1 + 
         p[7] * rbf2 +
         p[8] * sx + 
         p[9] * cx)
    
    return y
# EVOLVE-BLOCK-END
