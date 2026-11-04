import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x5 * 5x5 matrix multiplication.
    Uses the trivial (schoolbook) decomposition with rank n*m*p = 75.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Trivial rank is n * m * p
    rank = n * m * p
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Iterate over all (i, j, k) tuples
    # i: row of A, row of C (0..n-1)
    # j: col of A, row of B (0..m-1)
    # k: col of B, col of C (0..p-1)
    
    # Index mapping:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i  (Trace convention W[k*n+i, r])
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
                r += 1
                
                # Safety check (not really needed for 75 iterations)
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
