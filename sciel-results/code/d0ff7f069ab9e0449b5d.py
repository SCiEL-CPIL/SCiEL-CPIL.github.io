"""
Symbolic regression with a hybrid Fourier + RBF basis.
The problem is linear in parameters. We use a combination of:
- Fourier terms (cos and sin) for global smooth, oscillatory structure
- Gaussian RBFs for local features, kinks, or non-smooth behavior

This hybrid approach:
- Fourier terms capture periodic or smooth global trends
- RBF terms capture local deviations, discontinuities, or sharp features
- The combination is linear in params, so BFGS converges to the global optimum
- Well-conditioned and stable for optimization
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a hybrid Fourier + RBF basis.
    
    Parameters:
    x: Input array of shape (N,).
    params: Array of 10 coefficients.
        - params[0:6]: 6 Fourier terms (DC + cos/sin for freq 1-2)
        - params[6:10]: 4 Gaussian RBF terms
    
    Returns:
    Prediction array of shape (N,).
    """
    # Scale x from [-2, 2] to [-pi, pi] for standard Fourier frequencies
    t = x * (np.pi / 2.0)
    
    result = np.zeros_like(x, dtype=np.float64)
    
    # Fourier component (params[0:6])
    # DC term
    result += params[0]
    
    # Frequency 1
    result += params[1] * np.cos(t)
    result += params[2] * np.sin(t)
    
    # Frequency 2
    t2 = 2.0 * t
    result += params[3] * np.cos(t2)
    result += params[4] * np.sin(t2)
    
    # One more cosine term at frequency 3 for extra global structure
    t3 = 3.0 * t
    result += params[5] * np.cos(t3)
    
    # RBF component (params[6:10])
    # Fixed centers at -1.5, -0.5, 0.5, 1.5
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    sigma = 0.7  # Moderate width for locality
    
    for i in range(4):
        diff = x - centers[i]
        rbf = np.exp(-0.5 * (diff / sigma) ** 2)
        result += params[6 + i] * rbf
    
    return result
