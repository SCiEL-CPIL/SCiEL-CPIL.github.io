# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x4 matrix.
    This implementation uses the Schoolbook method, which yields a rank of n*m*p = 48.
    While not the theoretical minimum, it is a valid, exact, and robust baseline.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = 3, 4, 4
    rank = n * m * p  # 48
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U is indexed by [i*m + j]
                # V is indexed by [j*p + k]
                # W is indexed by [k*n + i] (Trace convention: W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
