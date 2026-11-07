import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a full cubic polynomial
    basis in position (x) and velocity (v).
    This approach uses a dense expansion up to degree 3, allowing for even-order
    terms and cross-couplings that the sparse physical model might miss.
    """
    # Scale inputs to normalize magnitude and improve conditioning
    # x range approx [-0.6, 0.6], v range approx [-0.3, 0.3]
    X = x / 0.6
    V = v / 0.3
    
    # Full cubic polynomial basis (10 terms)
    # Order: 1, X, V, X^2, XV, V^2, X^3, X^2V, XV^2, V^3
    b0 = 1.0
    b1 = X
    b2 = V
    b3 = X * X
    b4 = X * V
    b5 = V * V
    b6 = b3 * X
    b7 = b3 * V
    b8 = X * b5
    b9 = b5 * V
    
    a = (params[0] * b0
         + params[1] * b1
         + params[2] * b2
         + params[3] * b3
         + params[4] * b4
         + params[5] * b5
         + params[6] * b6
         + params[7] * b7
         + params[8] * b8
         + params[9] * b9)
         
    return a
