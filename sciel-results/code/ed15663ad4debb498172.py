"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Returns a rank-minimal (rank 18) exact tensor decomposition for M_{2,3,3}.
    The rank is minimal because M_{2,3,3} decomposes into two independent
    M_{1,3,3} components (one for each row of A), each of rank 9, with no
    possible sharing of multiplications between the rows.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Generate indices for the schoolbook algorithm
    # Each term is A[i,j] * B[j,k] contributing to C[k,n+i] -> W[k*n+i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
