"""
Symbolic regression with a Fourier series basis.
The problem is linear in parameters. We use a Fourier basis scaled to the domain [-2, 2].
This provides a global, smooth, orthogonal representation that is well-conditioned for BFGS.
"""
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a 10-term Fourier series.
    
    Parameters:
    x: Input array of shape (N,).
    params: Array of 10 coefficients.
    
    Returns:
    Prediction array of shape (N,).
    """
    # Scale x from [-2, 2] to [-pi, pi] for standard Fourier frequencies
    # t = x * (pi / 2)
    t = x * (np.pi / 2.0)
    
    # Precompute sine and cosine values for frequencies 1 to 5
    # We have 10 params:
    # 0: Constant (DC)
    # 1: cos(t)
    # 2: sin(t)
    # 3: cos(2t)
    # 4: sin(2t)
    # 5: cos(3t)
    # 6: sin(3t)
    # 7: cos(4t)
    # 8: sin(4t)
    # 9: cos(5t)
    
    result = np.zeros_like(x, dtype=np.float64)
    
    # DC term
    result += params[0]
    
    # Frequency 1
    result += params[1] * np.cos(t)
    result += params[2] * np.sin(t)
    
    # Frequency 2
    t2 = 2.0 * t
    result += params[3] * np.cos(t2)
    result += params[4] * np.sin(t2)
    
    # Frequency 3
    t3 = 3.0 * t
    result += params[5] * np.cos(t3)
    result += params[6] * np.sin(t3)
    
    # Frequency 4
    t4 = 4.0 * t
    result += params[7] * np.cos(t4)
    result += params[8] * np.sin(t4)
    
    # Frequency 5 (Cosine only to fit 10 params)
    t5 = 5.0 * t
    result += params[9] * np.cos(t5)
    
    return result
