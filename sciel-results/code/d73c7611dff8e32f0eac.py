"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm, mp, pn = n * m, m * p, p * n  # 16, 36, 36
    
    # Precompute support indices
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for idx, (a, b, c) in enumerate(support):
            U[a, idx] = 1
            V[b, idx] = 1
            W[c, idx] = 1
        return U, V, W, rank
    
    # Return schoolbook decomposition immediately as it is always valid and fast.
    # This ensures we do not timeout and provide a valid baseline score.
    U, V, W, rank = schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
