"""Matrix multiplication 4x5, 5x5 decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    # Use the standard rank-n*p decomposition for C[i,k] = sum_j A[i,j]*B[j,k]
    # This gives rank = n*p = 20.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Indexing convention:
    #   U row: i*m + j  (for A[i,j])
    #   V row: j*p + k  (for B[j,k])
    #   W row: k*n + i  (for C[i,k], trace convention)
    
    rank = n * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each (i, k) pair, create a rank-1 term
    # The term corresponds to summing over j: A[i,j] * B[j,k] contributes to C[i,k]
    for i in range(n):
        for k in range(p):
            r = i * p + k  # rank index for this (i,k) term
            
            # U: A[i,j] for all j
            for j in range(m):
                U[i * m + j, r] = 1.0
            
            # V: B[j,k] for all j
            for j in range(m):
                V[j * p + k, r] = 1.0
            
            # W: C[i,k] gets the sum
            W[k * n + i, r] = 1.0
    
    # Convert to list of lists with integer/half-integer values
    def to_list(mat):
        return [[int(round(val)) if abs(val - round(val)) < 1e-10 else val 
                 for val in row] for row in mat.tolist()]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
