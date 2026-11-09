"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 60 decomposition
    # The tensor rank of M_{2,3,10} is 60.
    # It is the direct sum of 10 tensors M_{2,3,1}, each of rank 6.
    rank = n * m * p  # 2 * 3 * 10 = 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U row i * m + k corresponds to A[i, k]
    # V row k * p + j corresponds to B[k, j]
    # W row j * n + i corresponds to C[i, j] (based on W[k*n+i] convention where k is col index j)
    
    for j in range(p):
        for k in range(m):
            for i in range(n):
                # This loop generates all 60 terms.
                # Term: A[i, k] * B[k, j] -> C[i, j]
                col = j * (n * m) + k * n + i
                # Wait, the column index in U, V, W must be the same 'r'.
                # We can map r to (j, k, i) uniquely.
                # Total 60 columns.
                r = j * 6 + k * 2 + i
                
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
