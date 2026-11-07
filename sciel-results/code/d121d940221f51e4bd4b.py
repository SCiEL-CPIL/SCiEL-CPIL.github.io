"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters.
    
    Strategy: Use a flexible basis that can represent both linear and 
    sinusoidal/nonlinear functions. The model is:
      p0 * x + p1 * y + p2 + p3 * sin(p4*x + p5*y + p6) + p7 * cos(p8*x - p9*y)
    
    This allows BFGS to jointly optimize amplitudes, frequencies, and phases.
    For purely linear targets, the sin/cos terms can be driven to zero amplitude.
    For sinusoidal targets like sin(x+y), BFGS can find p4=1, p5=1, p6=0, p3=1.
    """
    # Ensure params has at least 10 elements
    if len(params) < 10:
        # Pad with zeros if needed (shouldn't happen per contract)
        p = np.zeros(10)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    # Linear part
    pred = p[0] * x + p[1] * y + p[2]
    
    # Sinusoidal part: p3 * sin(p4*x + p5*y + p6)
    arg_sin = p[4] * x + p[5] * y + p[6]
    pred = pred + p[3] * np.sin(arg_sin)
    
    # Cosine part: p7 * cos(p8*x - p9*y)
    arg_cos = p[8] * x - p[9] * y
    pred = pred + p[7] * np.cos(arg_cos)
    
    return pred
# EVOLVE-BLOCK-END
