"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct the tensor decomposition for 3x5 by 5x6 matrix multiplication.
    Uses a vectorized Kronecker product approach to build the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank = n * m * p
    
    # Create basis vectors for the dimensions
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    
    # We can construct this by iterating over the non-zero entries of the tensor
    # which correspond to the identity mapping in the multiplication.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the matrices
    # The tensor T has T[i,j,k] = 1 for the multiplication C[i,k] += A[i,j] * B[j,k]
    # In the decomposition, each term r corresponds to a specific (i,j,k)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
