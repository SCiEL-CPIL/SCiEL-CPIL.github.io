import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a target function using a full degree-3 polynomial basis in x and y.
    This utilizes all 10 parameters to form a linear combination of basis functions:
    1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3.
    
    This design is convex with respect to the parameters, allowing BFGS to reliably
    find the global minimum for the training MSE, avoiding local minima issues
    associated with non-linear neural network structures.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Basis functions
    # Degree 0
    b0 = np.ones_like(x)
    # Degree 1
    b1 = x
    b2 = y
    # Degree 2
    b3 = x * x
    b4 = x * y
    b5 = y * y
    # Degree 3
    b6 = x * x * x
    b7 = x * x * y
    b8 = x * y * y
    b9 = y * y * y
    
    # Linear combination
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + p4 * b4 + p5 * b5 + p6 * b6 + p7 * b7 + p8 * b8 + p9 * b9
