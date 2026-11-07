"""Rewritten program: multi-basis symbolic regression with adaptive parameter allocation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multi-basis model: combines Chebyshev, Fourier, and localized bump functions.
    params[0:6] - Chebyshev coefficients (T_0..T_5)
    params[6:8] - Fourier (sin, cos) coefficients
    params[8]   - amplitude of a localized bump
    params[9]   - center of the localized bump (in [-2,2])
    
    Actually, let's use all 10 params more flexibly:
    params[0:4] - Chebyshev T_0..T_3
    params[4:6] - sin(x), cos(x)
    params[6:8] - tanh(x), x^2 (or similar)
    params[8:10] - localized Gaussian bump: amplitude and center
    """
    t = x / 2.0  # Map to [-1, 1]
    
    # Chebyshev polynomials T_0..T_3
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    
    # Fourier components
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    
    # Nonlinear activations
    tanh_x = np.tanh(x)
    x2 = x**2
    
    # Localized Gaussian bump: amplitude * exp(-((x - center)/width)^2)
    # Use a fixed width of 0.5 for stability, or make it part of params
    # Let's use params[8] as amplitude, params[9] as center
    amplitude = params[8]
    center = params[9]
    width = 0.5  # Fixed width for stability
    bump = amplitude * np.exp(-((x - center) / width)**2)
    
    # Combine all basis functions
    result = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * T3 +
              params[4] * sin_x + 
              params[5] * cos_x +
              params[6] * tanh_x + 
              params[7] * x2 +
              bump)
    
    return result
# EVOLVE-BLOCK-END
