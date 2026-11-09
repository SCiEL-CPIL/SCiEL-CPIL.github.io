import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear damped oscillator with periodic driving.
    Uses all 10 parameters to construct a rich nonlinear model.
    
    Model: a = c0 + c1*t + c2*x + c3*v + c4*x^2 + c5*v^2 + c6*x*v + c7*sin(c8*t + c9*x)
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Linear terms
    term_linear = c0 + c1 * t + c2 * x + c3 * v
    
    # Quadratic/nonlinear terms
    term_quad = c4 * x**2 + c5 * v**2 + c6 * x * v
    
    # Periodic driving term with phase modulation
    # sin(c8*t + c9*x) provides a nonlinear coupling between time and position
    term_drive = c7 * np.sin(c8 * t + c9 * x)
    
    return term_linear + term_quad + term_drive
