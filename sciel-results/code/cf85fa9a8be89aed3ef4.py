import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 10 decomposition based on decoupling rows
    # Each row (1x5) x (5x5) can be done in rank 5
    # Total rank = n * 5 = 10
    rank = n * 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i in 0..n-1
    for i in range(n):
        # Assign rank block of size 5 for this row
        start_r = i * 5
        for r in range(5):
            col = start_r + r
            
            # U indices: i*m + j
            # We set U[i*m + r, col] = 1
            # This corresponds to j = r
            U[i * m + r][col] = 1
            
            # V indices: j*p + k
            # We set V[j*p + k, col] = 1 if j == r
            # So for j=r, all k, V[r*p + k, col] = 1
            for k in range(p):
                V[r * p + k][col] = 1
                
            # W indices: k*n + i
            # We set W[k*n + i, col] = 1 for all k
            for k in range(p):
                W[k * n + i][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
