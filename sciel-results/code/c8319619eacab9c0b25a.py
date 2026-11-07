"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of Tanh non-linearities and a linear term.
    y = m*x + sum(tanh(w_i*x + b_i)) + b_out
    
    Parameters (10 total):
    p[0]: m (linear slope)
    p[1..4]: w_1..w_4 (tanh weights)
    p[5..8]: b_1..b_4 (tanh biases)
    p[9]: b_out (output bias)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    m = params[0]
    w1, w2, w3, w4 = params[1], params[2], params[3], params[4]
    b1, b2, b3, b4 = params[5], params[6], params[7], params[8]
    b_out = params[9]
    
    # Compute non-linear terms
    # Using tanh which is bounded and smooth
    term1 = np.tanh(w1 * x + b1)
    term2 = np.tanh(w2 * x + b2)
    term3 = np.tanh(w3 * x + b3)
    term4 = np.tanh(w4 * x + b4)
    
    # Linear term
    lin = m * x
    
    # Sum all components
    y = lin + term1 + term2 + term3 + term4 + b_out
    
    return y
# EVOLVE-BLOCK-END
