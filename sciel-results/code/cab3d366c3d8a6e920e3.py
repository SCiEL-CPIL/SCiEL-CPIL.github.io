import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a complete cubic polynomial basis.
    Inputs are scaled to improve numerical conditioning for the optimizer.
    """
    # Scale inputs to roughly [-1, 1] range to help BFGS
    # x range approx [-0.57, 0.53], v range approx [-0.18, 0.28]
    X = x / 0.6
    V = v / 0.3
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    term_0 = p0
    term_1 = p1 * X
    term_2 = p2 * V
    term_3 = p3 * X**2
    term_4 = p4 * X * V
    term_5 = p5 * V**2
    term_6 = p6 * X**3
    term_7 = p7 * X**2 * V
    term_8 = p8 * X * V**2
    term_9 = p9 * V**3
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
