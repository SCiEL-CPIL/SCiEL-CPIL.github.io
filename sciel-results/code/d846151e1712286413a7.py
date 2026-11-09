import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator: a = f(x, v, t)
    
    Uses a combination of polynomial terms in x and v, and Fourier terms in t.
    Normalization is applied to x and v to improve conditioning for BFGS.
    
    Parameters (10 total):
    p0: bias
    p1: linear x coefficient
    p2: cubic x coefficient (Duffing)
    p3: linear v coefficient (damping)
    p4: cubic v coefficient (nonlinear damping)
    p5: x*v coupling coefficient
    p6: sin(t) forcing
    p7: cos(t) forcing
    p8: sin(2t) forcing
    p9: cos(2t) forcing
    """
    # Normalize x and v to roughly [-1, 1] range to help BFGS
    # x range approx [-0.2, 0.17], scale 0.2, center -0.015
    # v range approx [-0.34, 0.36], scale 0.35, center 0.01
    
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x_n * x_n
    x3 = x2 * x_n
    v2 = v_n * v_n
    v3 = v2 * v_n
    xv = x_n * v_n
    
    st = np.sin(t)
    ct = np.cos(t)
    st2 = np.sin(2.0 * t)
    ct2 = np.cos(2.0 * t)
    
    # Linear combination
    res = (p0 + 
           p1 * x_n + 
           p2 * x3 + 
           p3 * v_n + 
           p4 * v3 + 
           p5 * xv + 
           p6 * st + 
           p7 * ct + 
           p8 * st2 + 
           p9 * ct2)
           
    return res
