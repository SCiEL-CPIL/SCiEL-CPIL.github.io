"""Construct a rank-18 tensor decomposition for 3x3 * 3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    # Rank-18 decomposition: for each (i, k) pair (12 pairs), one rank-1 term
    # that computes C[i,k] = sum_j A[i,j] * B[j,k]
    
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]   # 9 rows: indexed by (i, j)
    V = [[0] * rank for _ in range(m * p)]   # 12 rows: indexed by (j, k)
    W = [[0] * rank for _ in range(p * n)]   # 12 rows: indexed by (k, i)
    
    # For each (i0, k0) pair, create one term r
    # This term contributes to C[i0, k0] by summing over j: A[i0, j] * B[j, k0]
    # 
    # Term r for (i0, k0):
    #   U[i0, j, r] = 1 for all j (picks up A[i0, j])
    #   V[j, k0, r] = 1 for all j (picks up B[j, k0])
    #   W[k0, i0, r] = 1 (routes to output C[i0, k0])
    #
    # Verification:
    # C[i, k] = sum_r U[i, j', r] * V[j'', k'', r] * W[k, i', r] * A[i', j'] * B[j'', k'']
    # For term r = (i0, k0):
    #   U[i, j', r] = delta_{i, i0} (for all j')
    #   V[j'', k'', r] = delta_{k'', k0} (for all j'')
    #   W[k, i', r] = delta_{k, k0} * delta_{i', i0}
    # 
    # So the tensor coefficient for term r is:
    #   delta_{i, i0} * delta_{k'', k0} * delta_{k, k0} * delta_{i', i0}
    #   = delta_{i, i0} * delta_{k, k0} * delta_{k'', k0} * delta_{i', i0}
    #
    # C[i, k] = sum_{i0, k0} delta_{i, i0} delta_{k, k0} delta_{k'', k0} delta_{i', i0} A[i', j'] B[j'', k'']
    #          = A[i, j'] B[j'', k] summed appropriately
    #          = sum_{j'} A[i, j'] * B[j', k]  (when j'' = j' and k'' = k)
    
    r_idx = 0
    for i0 in range(n):
        for k0 in range(p):
            # Set U entries: U[i0, j, r] = 1 for j = 0, 1, 2
            for j in range(m):
                U[i0 * m + j][r_idx] = 1
            # Set V entries: V[j, k0, r] = 1 for j = 0, 1, 2
            for j in range(m):
                V[j * p + k0][r_idx] = 1
            # Set W entry: W[k0, i0, r] = 1
            W[k0 * n + i0][r_idx] = 1
            r_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
