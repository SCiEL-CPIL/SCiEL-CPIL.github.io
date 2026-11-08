import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator using a truncated Fourier series.
    
    The function approximates the target as a sum of sinusoidal terms in x and v.
    This basis is well-suited for oscillatory systems where the relationship
    between state (x, v) and acceleration/force is periodic.
    
    f(x, v) = p0 
             + p1*sin(x) + p2*cos(x)
             + p3*sin(v) + p4*cos(v)
             + p5*sin(x+v) + p6*cos(x+v)
             + p7*sin(x-v) + p8*cos(x-v)
             + p9*sin(2*x)
             
    This uses 10 parameters to capture fundamental frequencies and simple harmonics.
    """
    p = params
    
    # Pre-compute trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Combined angles
    sin_xv = np.sin(x + v)
    cos_xv = np.cos(x + v)
    sin_xm = np.sin(x - v)
    cos_xm = np.cos(x - v)
    sin_2x = np.sin(2 * x)
    
    # Linear combination of basis functions
    # p[0] is the bias (DC component)
    result = (p[0] 
              + p[1] * sin_x 
              + p[2] * cos_x 
              + p[3] * sin_v 
              + p[4] * cos_v 
              + p[5] * sin_xv 
              + p[6] * cos_xv 
              + p[7] * sin_xm 
              + p[8] * cos_xm 
              + p[9] * sin_2x)
              
    return result
