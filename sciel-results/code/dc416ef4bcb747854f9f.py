"""Matrix multiplication 2,3,9 - rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor for A (n x m) and B (m x p) producing C (p x n)
    # can be decomposed into n independent blocks, one for each row i of A.
    # Each block corresponds to multiplying a 1 x m row vector by the m x p matrix B.
    # The rank of a 1 x m x p multiplication tensor is m * p.
    # Since the output rows of C are disjoint for different i, the total rank is the sum.
    # Rank = n * (m * p) = 2 * 3 * 9 = 54.
    
    rank_per_block = m * p
    total_rank = n * rank_per_block
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Construct each block independently
    for i in range(n):
        start_col = i * rank_per_block
        end_col = (i + 1) * rank_per_block
        
        # For this block (row i of A), we iterate over j (col of A / row of B) and k (col of B)
        for j in range(m):
            for k in range(p):
                col_idx = start_col + (j * p + k)
                
                # U entry: corresponds to A[i, j]
                # Index in U: i * m + j
                U[i * m + j, col_idx] = 1
                
                # V entry: corresponds to B[j, k]
                # Index in V: j * p + k
                V[j * p + k, col_idx] = 1
                
                # W entry: corresponds to C[k, i] (note: output is p x n, index k*n + i)
                # The task specifies W[k*n+i, r]
                W[k * n + i, col_idx] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
