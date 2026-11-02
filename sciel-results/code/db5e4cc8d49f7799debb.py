"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # The tensor for M_{n,m,p} is a direct sum of p copies of M_{n,m,1}.
    # The rank of M_{n,m,1} is n*m (trivial, and optimal for small n,m).
    # For n=2, m=2, rank is 4.
    # Total rank = p * n * m = 13 * 4 = 52.
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = n * m * p
    
    # Use lists of lists as required by the interface
    U = [[0.0] * rank for _ in range(dim_u)]
    V = [[0.0] * rank for _ in range(dim_v)]
    W = [[0.0] * rank for _ in range(dim_w)]
    
    # Fill the matrices
    # Column index corresponds to the triplet (k, i, j)
    # k in [0, p), i in [0, n), j in [0, m)
    for k in range(p):
        for i in range(n):
            for j in range(m):
                col = k * (n * m) + (i * m + j)
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
