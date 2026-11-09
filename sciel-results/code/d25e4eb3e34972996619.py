import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize as numpy arrays for efficient construction, then convert to lists
    # The evaluator expects {'U', 'V', 'W'} with lists of lists (or compatible numeric type)
    # The prompt says "integer/half-integer entries".
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: row i*m + j
                # Index in V: row j*p + k
                # Index in W: row k*n + i
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
                
    # Convert to lists to ensure strict interface compliance if needed
    # "factor matrices of shapes (n*m, rank), ..."
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
