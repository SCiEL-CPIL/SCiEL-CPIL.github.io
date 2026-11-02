"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for 3x5x7 matrix multiplication.
    Uses a Kronecker product structure to decouple the dimensions, representing
    the multiplication tensor as a sum of rank-one terms derived from the outer
    products of identity matrices.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We construct the U, V, W matrices using Kronecker products of identity matrices
    # to satisfy the structural requirement.
    
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Each column corresponds to a rank-1 term.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Construct u vector for A[i, j]
                u = np.zeros(n * m)
                u[i * m + j] = 1.0
                
                # Construct v vector for B[j, k]
                v = np.zeros(m * p)
                v[j * p + k] = 1.0
                
                # Construct w vector for C[i, k] (index k*n + i)
                w = np.zeros(p * n)
                w[k * n + i] = 1.0
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    # Stack columns into matrices
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Convert to list of lists to match the expected interface format
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
