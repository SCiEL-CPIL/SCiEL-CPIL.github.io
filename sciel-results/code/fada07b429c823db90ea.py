import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a fully flexible 4-term polynomial response surface.
    This replaces rigid physical kernels with a flexible quadratic interaction model.
    """
    # Unpack parameters
    # params[0] is the intercept (c0)
    # params[1..15] would be coefficients, but we only have 10 params.
    # So we need to be clever. 
    # Let's use a quadratic model for the 4 inputs.
    # Inputs: b, s, temp, pH
    # Terms: 1, b, s, temp, pH, b^2, s^2, temp^2, pH^2, b*s, b*temp, b*pH, s*temp, s*pH, temp*pH
    # That's 15 terms. We only have 10 params.
    # We must drop some terms or use a different structure.
    # The inspiration 1 mentions "quadratic interactions" and scored 1.0443.
    # Let's try a model with all main effects, all squared terms, and a few key interactions.
    # Or perhaps a simpler flexible model: 
    # y = c0 + c1*b + c2*s + c3*temp + c4*pH + c5*b^2 + c6*s^2 + c7*temp^2 + c8*pH^2 + c9*(b*s)
    # This uses all 10 params.
    
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
    
    # Compute terms
    term1 = c0
    term2 = c1 * b
    term3 = c2 * s
    term4 = c3 * temp
    term5 = c4 * pH
    term6 = c5 * (b ** 2)
    term7 = c6 * (s ** 2)
    term8 = c7 * (temp ** 2)
    term9 = c8 * (pH ** 2)
    term10 = c9 * (b * s)
    
    y = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
    
    return y
