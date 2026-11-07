import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Degree 9 Polynomial in scaled input.
    
    The 10 parameters correspond to the coefficients of a polynomial of degree 9.
    To ensure numerical stability and better conditioning for BFGS, we scale the
    input x from [-2, 2] to u in [-1, 1] by dividing by 2.
    
    y = c0 + c1*u + c2*u^2 + ... + c9*u^9
    
    This is a universal approximator for continuous functions on the interval.
    It differs from the current Gaussian+Poly approach by using a purely global
    basis set, avoiding the local minima pitfalls of learning Gaussian centers
    from an initial guess of 1.0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Scale input to [-1, 1] for better polynomial conditioning
    u = x / 2.0
    
    # Extract coefficients
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Evaluate polynomial using Horner's method for efficiency and stability
    # y = c9*u^9 + c8*u^8 + ... + c1*u + c0
    y = c9
    y = y * u + c8
    y = y * u + c7
    y = y * u + c6
    y = y * u + c5
    y = y * u + c4
    y = y * u + c3
    y = y * u + c2
    y = y * u + c1
    y = y * u + c0
    
    return y
