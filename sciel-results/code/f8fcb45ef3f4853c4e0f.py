import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The target tensor is the all-ones tensor of size n x m x p
    # T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # This corresponds to the trace form sum(A_ij * B_jk * C_ki)
    # The rank of the all-ones tensor is 1.
    # We can construct a rank-1 decomposition with U, V, W all being vectors of ones.
    
    rank = 1
    
    # U has shape (n*m, rank)
    U = [[1] for _ in range(n * m)]
    
    # V has shape (m*p, rank)
    V = [[1] for _ in range(m * p)]
    
    # W has shape (p*n, rank)
    # Note: The prompt specifies W[k*n+i, r].
    # In the trace sum sum_i,j,k A_ij B_jk C_ki, the output index is (k,i) or (i,k)?
    # The seed code uses W[k*n+i].
    # If we set W to all ones, the specific indexing order doesn't affect the value, only the shape.
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
