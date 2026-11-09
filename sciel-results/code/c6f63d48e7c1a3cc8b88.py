import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of a complete cubic polynomial basis.
    
    Basis functions:
    1. 1
    2. x
    3. y
    4. x^2
    5. y^2
    6. xy
    7. x^3
    8. y^3
    9. x^2 * y
    10. x * y^2
    
    The model is:
    f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*y^2 + p5*xy + p6*x^3 + p7*y^3 + p8*x^2*y + p9*x*y^2
    """
    # Ensure inputs are arrays for vectorized operations
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Calculate basis terms
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    term6 = term3 * x
    term7 = term4 * y
    term8 = term3 * y
    term9 = term4 * x
    
    # Linear combination
    pred = (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
            
    return pred
