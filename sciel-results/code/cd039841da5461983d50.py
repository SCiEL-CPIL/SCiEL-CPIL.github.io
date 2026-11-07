import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression model using a fixed library of basis functions.
    The model is linear in the parameters, allowing BFGS to efficiently
    find the global optimum for the given basis set.
    
    Basis functions:
    1. 1 (Bias)
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. sin(x)
    8. sin(y)
    9. cos(x)
    10. cos(y)
    """
    N = len(x)
    # Pre-allocate design matrix for efficiency
    X = np.empty((N, 10), dtype=np.float64)
    
    X[:, 0] = 1.0
    X[:, 1] = x
    X[:, 2] = y
    X[:, 3] = x * x
    X[:, 4] = y * y
    X[:, 5] = x * y
    X[:, 6] = np.sin(x)
    X[:, 7] = np.sin(y)
    X[:, 8] = np.cos(x)
    X[:, 9] = np.cos(y)
    
    return X @ params
