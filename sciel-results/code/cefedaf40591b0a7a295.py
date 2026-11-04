import numpy as np
import time

_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def construct():
    if not _time_ok():
        pass

    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen factors for 2x2 matrix multiplication
    # Shapes: (4, 7)
    # Rows correspond to flattened indices of A, B, C respectively.
    # A: 0=A00, 1=A01, 2=A10, 3=A11
    # B: 0=B00, 1=B01, 2=B10, 3=B11
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 1, 0, 1, 0]
    ])
    
    # Construct 4x4 factors using Kronecker product
    # This ensures correct index mapping for the standard tensor product
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    # Convert to lists as required by the previous interface style
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
