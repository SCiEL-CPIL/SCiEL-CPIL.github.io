import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter model combining polynomial and trigonometric basis functions.
    
    params:
      0: a0 (constant)
      1: a1 (x)
      2: a2 (x^2)
      3: a3 (x^3)
      4: a4 (x^4)
      5: b1 (sin(x))
      6: b2 (cos(x))
      7: b3 (sin(2x))
      8: b4 (cos(2x))
      9: c1 (tanh(x))
    
    The model is:
      f(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
             + b1*sin(x) + b2*cos(x) + b3*sin(2x) + b4*cos(2x)
             + c1*tanh(x)
    
    This is a linear model in params, so the optimization landscape is convex
    and BFGS should converge reliably. The basis covers polynomial, periodic,
    and saturation features simultaneously.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Extract parameters
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    a4 = params[4]
    b1 = params[5]
    b2 = params[6]
    b3 = params[7]
    b4 = params[8]
    c1 = params[9]
    
    # Compute polynomial part
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4
    
    # Compute trigonometric part
    trig = b1 * np.sin(x) + b2 * np.cos(x) + b3 * np.sin(2 * x) + b4 * np.cos(2 * x)
    
    # Compute tanh part
    tanh_part = c1 * np.tanh(x)
    
    result = poly + trig + tanh_part
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
