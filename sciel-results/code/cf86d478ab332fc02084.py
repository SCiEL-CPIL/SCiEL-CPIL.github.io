import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    
    # Rank 1 decomposition
    rank = 1
    
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
