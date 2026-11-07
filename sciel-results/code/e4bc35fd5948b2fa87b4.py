import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using scaled monomial polynomial basis.
    Inputs are mapped to [-1, 1] to improve conditioning.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Map x to [-1, 1]
    x_center = -0.02265
    x_scale = 0.55015
    X = (x - x_center) / x_scale
    
    # Map v to [-1, 1]
    v_center = 0.05335
    v_scale = 0.22875
    V = (v - v_center) / v_scale
    
    # Complete polynomial basis up to degree 3 in X and V
    term0 = 1.0
    term1 = X
    term2 = V
    term3 = X * X
    term4 = X * V
    term5 = V * V
    term6 = X * X * X
    term7 = X * X * V
    term8 = X * V * V
    term9 = V * V * V
    
    return p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9
