"""Matrix multiplication 3,3,7 - rank-minimal decomposition using tensor network structure."""

import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x7 matrix.
    
    The multiplication tensor T has entries T[i,j,k] where:
      - i indexes rows of A (size n=3)
      - j indexes columns of A / rows of B (size m=3)  
      - k indexes columns of B (size p=7)
    
    Output C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using the trace convention: W[k*n+i, r]
    
    The schoolbook decomposition gives rank = n*m*p = 63.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # U: (n*m) x rank
    # V: (m*p) x rank  
    # W: (p*n) x rank
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Convert to lists of lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
