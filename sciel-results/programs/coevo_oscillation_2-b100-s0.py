import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a polynomial basis model.
    
    The acceleration y is modeled as a linear combination of polynomial terms
    in position x and velocity v, plus trigonometric terms in time t.
    This structure captures common nonlinear oscillator dynamics (e.g., Duffing-like)
    and periodic forcing while remaining well-conditioned for optimization.
    
    Inputs:
    t: time
    x: position
    v: velocity
    params: array of 10 fitted constants
    
    Returns:
    y: acceleration
    """
    # Scale inputs to O(1) to improve conditioning for BFGS
    # Based on training ranges: x ~ [-0.21, 0.17], v ~ [-0.34, 0.36]
    # t is in [30, 50], sin/cos handle it naturally, no scaling needed for trig
    x_scale = 0.2
    v_scale = 0.35
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Define basis functions
    # State terms
    b0 = xs
    b1 = vs
    b2 = xs * xs
    b3 = vs * vs
    b4 = xs * vs
    b5 = xs * xs * xs
    b6 = vs * vs * vs
    
    # Time terms
    st = np.sin(t)
    ct = np.cos(t)
    
    # Additional mixed or higher order terms to utilize all 10 params
    # p7, p8 for forcing, p9 for bias or another interaction
    # Let's use p7*st + p8*ct + p9*1.0 (bias)
    # Or maybe p9*xs*st? Let's stick to bias for stability, or x^2*vs?
    # A bias term is often helpful.
    b7 = st
    b8 = ct
    b9 = np.ones_like(t)
    
    # Construct the prediction
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
        
    return result
