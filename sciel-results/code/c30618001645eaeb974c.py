"""Matrix multiplication 2x3x3 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    rows_U = n * m
    rows_V = m * p
    rows_W = p * n
    
    # Initialize with zeros
    U = [[0.0] * R for _ in range(rows_U)]
    V = [[0.0] * R for _ in range(rows_V)]
    W = [[0.0] * R for _ in range(rows_W)]
    
    # Iterate over all combinations of j, k, i to fill the matrices
    # We assign a unique rank index r to each (j, k, i) tuple
    # Note: The mapping of (i, j, k) to r must be consistent.
    # Let r = j * (p * n) + k * n + i
    # This covers all 18 combinations uniquely.
    
    idx = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                # U index: i * m + j
                u_row = i * m + j
                # V index: j * p + k
                v_row = j * p + k
                # W index: k * n + i
                w_row = k * n + i
                
                U[u_row][idx] = 1.0
                V[v_row][idx] = 1.0
                W[w_row][idx] = 1.0
                idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
