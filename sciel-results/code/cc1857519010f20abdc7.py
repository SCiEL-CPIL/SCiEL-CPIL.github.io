import numpy as np
import time
import sys
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for n x m by m x p is the all-ones tensor
    # because every combination of (i, j, k) corresponds to a term A_ij * B_jk
    # contributing to C_ik with coefficient 1.
    # The rank of the all-ones tensor is 1.
    
    rank = 1
    
    U = np.ones((n * m, rank), dtype=np.float64)
    V = np.ones((m * p, rank), dtype=np.float64)
    W = np.ones((p * n, rank), dtype=np.float64)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
