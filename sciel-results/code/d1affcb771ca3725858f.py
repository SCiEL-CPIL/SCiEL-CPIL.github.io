"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using an Additive Polynomial Approach.
    
    This model uses a linear combination of polynomial basis functions to capture
    non-linearities and interactions without assuming a specific multiplicative
    mechanistic structure.
    
    Inputs are centered and scaled to [~ -1, 1] range to improve conditioning
    for BFGS optimization.
    
    Parameters (10 total):
    p0: Intercept
    p1: Linear coeff for b
    p2: Linear coeff for s
    p3: Linear coeff for temp
    p4: Linear coeff for pH
    p5: Quadratic coeff for temp
    p6: Quadratic coeff for pH
    p7: Interaction coeff for temp * pH
    p8: Interaction coeff for b * s
    p9: Cubic coeff for temp
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to centered range approx [-1, 1]
    # b in [1, 5] -> (b-3)/2 in [-1, 1]
    x_b = (b - 3.0) / 2.0
    # s in [1, 5] -> (s-3)/2 in [-1, 1]
    x_s = (s - 3.0) / 2.0
    # temp in [1, 40] -> (temp-20)/10 in [-1.9, 2.0]
    x_t = (temp - 20.0) / 10.0
    # pH in [2, 12] -> (pH-7)/5 in [-1, 1]
    x_p = (pH - 7.0) / 5.0
    
    # Construct basis terms
    term_0 = np.ones_like(b)
    term_1 = x_b
    term_2 = x_s
    term_3 = x_t
    term_4 = x_p
    term_5 = x_t ** 2
    term_6 = x_p ** 2
    term_7 = x_t * x_p
    term_8 = x_b * x_s
    term_9 = x_t ** 3
    
    # Linear combination
    result = (p0 * term_0 + p1 * term_1 + p2 * term_2 + p3 * term_3 + p4 * term_4 +
              p5 * term_5 + p6 * term_6 + p7 * term_7 + p8 * term_8 + p9 * term_9)
    
    return result
# EVOLVE-BLOCK-END
