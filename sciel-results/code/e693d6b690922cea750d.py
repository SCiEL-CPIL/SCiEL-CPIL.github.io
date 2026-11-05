import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 1
    
    U = np.ones((n * m, rank), dtype=float)
    V = np.ones((m * p, rank), dtype=float)
    W = np.ones((p * n, rank), dtype=float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
