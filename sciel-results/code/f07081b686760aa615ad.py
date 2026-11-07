"""Symbolic regression with a Fourier basis including a DC term."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with a DC (constant) term.
    
    y = c0 
      + c1*cos(pi*x) + c2*sin(pi*x)
      + c3*cos(2*pi*x) + c4*sin(2*pi*x)
      + c5*cos(3*pi*x) + c6*sin(3*pi*x)
      + c7*cos(4*pi*x) + c8*sin(4*pi*x)
      + c9*cos(5*pi*x)
      
    params layout (length 10):
      params[0] = c0 (DC term)
      params[1], params[2] = c1, c2 (freq 1)
      params[3], params[4] = c3, c4 (freq 2)
      params[5], params[6] = c5, c6 (freq 3)
      params[7], params[8] = c7, c8 (freq 4)
      params[9] = c9 (freq 5, cosine only)
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(params, dtype=float)
    
    # Precompute trigonometric bases
    # Using pi * x to match the parent's frequency scale
    # Frequencies: 1, 2, 3, 4, 5
    
    cos_1 = np.cos(np.pi * x)
    sin_1 = np.sin(np.pi * x)
    
    cos_2 = np.cos(2.0 * np.pi * x)
    sin_2 = np.sin(2.0 * np.pi * x)
    
    cos_3 = np.cos(3.0 * np.pi * x)
    sin_3 = np.sin(3.0 * np.pi * x)
    
    cos_4 = np.cos(4.0 * np.pi * x)
    sin_4 = np.sin(4.0 * np.pi * x)
    
    cos_5 = np.cos(5.0 * np.pi * x)
    
    # Construct the prediction
    # Ensure we handle the array shapes correctly
    result = (p[0] 
              + p[1] * cos_1 + p[2] * sin_1
              + p[3] * cos_2 + p[4] * sin_2
              + p[5] * cos_3 + p[6] * sin_3
              + p[7] * cos_4 + p[8] * sin_4
              + p[9] * cos_5)
    
    # Safety: guarantee finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
