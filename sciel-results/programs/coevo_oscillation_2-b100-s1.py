import numpy as np
import time

# EVOLVE-BLOCK-START

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with scaled inputs for better optimization conditioning.
    
    Model:
    a = p0 + p1*(t/50) + p2*(x/0.2) + p3*(v/0.3) + p4*(x/0.2)^2 + 
        p5*(v/0.3)^2 + p6*(x/0.2)*(v/0.3) + p7*(x/0.2)^3 + p8*(v/0.3)^3 + p9*sin(t)
    """
    # Deadline guard is not strictly necessary for a simple forward pass, 
    # but we ensure we don't hang if something weird happens with inputs.
    
    # Scale inputs to O(1) magnitude to help BFGS starting from ones
    T = t / 50.0
    X = x / 0.2
    V = v / 0.3
    
    # Compute basis terms
    term_0 = np.ones_like(t)
    term_1 = T
    term_2 = X
    term_3 = V
    term_4 = X * X
    term_5 = V * V
    term_6 = X * V
    term_7 = X * X * X
    term_8 = V * V * V
    term_9 = np.sin(t)
    
    # Combine with parameters
    pred = (params[0] * term_0 + 
            params[1] * term_1 + 
            params[2] * term_2 + 
            params[3] * term_3 + 
            params[4] * term_4 + 
            params[5] * term_5 + 
            params[6] * term_6 + 
            params[7] * term_7 + 
            params[8] * term_8 + 
            params[9] * term_9)
            
    return pred

# EVOLVE-BLOCK-END
