import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    U = np.ones((n * m, 1))
    V = np.ones((m * p, 1))
    W = np.ones((p * n, 1))
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
