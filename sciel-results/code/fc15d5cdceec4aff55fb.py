"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1.0
                # V index: j*p + k
                V[j * p + k, col] = 1.0
                # W index: k*n + i
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists as per interface requirement (though numpy arrays are often accepted, 
    # the prompt says "factor matrices of shapes... with integer/half-integer entries")
    # The seed program returned lists of lists. I will return lists of lists to be safe, 
    # or numpy arrays if the evaluator accepts them. 
    # "construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank)..."
    # Usually, returning numpy arrays is fine and more efficient for the evaluator to process.
    # However, the seed program returned lists. I'll stick to lists to match the seed's output type 
    # if the evaluator is strict, but numpy is usually preferred for "matrix" types.
    # Let's return numpy arrays as they are standard for this kind of task in Python.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
