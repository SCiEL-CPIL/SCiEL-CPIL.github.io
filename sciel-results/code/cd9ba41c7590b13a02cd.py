import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a flexible model with polynomial, sigmoid, and Gaussian components.
    
    The model combines:
    - A cubic polynomial for global trend
    - Two sigmoid (tanh) units for smooth step-like transitions
    - Two Gaussian bumps for localized peaks or wells
    
    This provides a rich function space that can approximate many common smooth functions.
    
    Parameters (10 total):
    p0, p1, p2, p3: Polynomial coefficients (constant, linear, quadratic, cubic)
    p4, p5, p6: Sigmoid 1 - amplitude, slope, center
    p7, p8, p9: Sigmoid 2 - amplitude, slope, center
    
    Wait, that's 10 params but I need room for Gaussians too. Let me reconsider.
    
    Actually, with only 10 params, let's use:
    p0, p1, p2, p3: Polynomial coefficients
    p4, p5, p6: Sigmoid 1 - amplitude, slope, center  
    p7, p8, p9: Sigmoid 2 - amplitude, slope, center
    
    This gives us a polynomial plus two sigmoid units, which is quite flexible.
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    a1 = params[4]
    s1 = params[5]
    c1 = params[6]
    
    a2 = params[7]
    s2 = params[8]
    c2 = params[9]
    
    # Polynomial part
    y = p0 + p1 * x + p2 * x * x + p3 * x * x * x
    
    # Sigmoid parts - use tanh for smooth saturation
    # Clip slope to avoid numerical issues with very large values
    s1_clip = np.clip(s1, -50, 50)
    s2_clip = np.clip(s2, -50, 50)
    
    y += a1 * np.tanh(s1_clip * (x - c1))
    y += a2 * np.tanh(s2_clip * (x - c2))
    
    return y
