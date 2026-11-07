import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic polynomial + 3 Gaussian RBFs with learnable centers and fixed widths.
    
    Model:
        y(x) = c0 + c1*x + c2*x^2 + c3*x^3 + sum_{j=1}^{3} a_j * exp(-w_j * (x - mu_j)^2)
    
    Parameters (10 total):
        params[0:4]  = c0, c1, c2, c3 (polynomial coefficients)
        params[4:7]  = a1, a2, a3 (RBF amplitudes)
        params[7:10] = mu1, mu2, mu3 (RBF centers, bounded via tanh to [-2, 2])
    
    Widths are fixed at [1.0, 0.5, 0.25] for stable optimization.
    Centers are bounded via tanh to keep them in a reasonable range.
    The polynomial part is linear in params (well-conditioned for BFGS).
    The RBF amplitudes are also linear. Only centers are nonlinear (via tanh).
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Polynomial part (linear in params)
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    y_poly = c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    # RBF part
    a1, a2, a3 = params[4], params[5], params[6]
    mu1 = 2.0 * np.tanh(params[7])
    mu2 = 2.0 * np.tanh(params[8])
    mu3 = 2.0 * np.tanh(params[9])
    
    # Fixed widths for stability
    w1, w2, w3 = 1.0, 0.5, 0.25
    
    y_rbf = (a1 * np.exp(-w1 * (x - mu1)**2) +
             a2 * np.exp(-w2 * (x - mu2)**2) +
             a3 * np.exp(-w3 * (x - mu3)**2))
    
    result = y_poly + y_rbf
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
