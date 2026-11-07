"""Symbolic regression with 1 input using a rich basis of functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points (200 train or 200 test)
    params: array of 10 parameters
    Returns: predicted values
    """
    n = len(x)
    
    # Build basis functions
    # We have 10 params. Let's use a mix of polynomial, trig, and rational-ish terms
    # to capture various functional forms.
    
    # 1. Constant
    b0 = np.ones(n)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic
    b2 = x**2
    
    # 4. Cubic
    b3 = x**3
    
    # 5. Sine
    b4 = np.sin(x)
    
    # 6. Cosine
    b5 = np.cos(x)
    
    # 7. Sine(2x)
    b6 = np.sin(2*x)
    
    # 8. Cosine(2x)
    b7 = np.cos(2*x)
    
    # 9. Tanh
    b8 = np.tanh(x)
    
    # 10. x * sin(x) or something non-linear? Let's try x * tanh(x) or just a higher poly?
    # Maybe x**4 is too high. Let's use x * sin(x) to capture oscillatory growth?
    # Or maybe just keep it simple with polynomials and trigs.
    # Let's use x * cos(x) for variety.
    b9 = x * np.cos(x)
    
    # Combine
    # y = p0*b0 + p1*b1 + ... + p9*b9
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
# EVOLVE-BLOCK-END
