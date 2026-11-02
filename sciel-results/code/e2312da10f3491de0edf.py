"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use the standard schoolbook algorithm for matrix multiplication.
    # The rank is n * m * p.
    # A is n x m (3x4), B is m x p (4x4), C is n x p (3x4).
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # A rank-1 term corresponds to a specific (i, j, k).
    # U is indexed by (i, j) -> i*m + j
    # V is indexed by (j, k) -> j*p + k
    # W is indexed by (k, i) -> k*n + i (following the trace convention W[k*n+i, r])
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
