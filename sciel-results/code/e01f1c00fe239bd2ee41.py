"""
Optimized program for matrix multiplication 2x3 * 3x9.
Uses a block-diagonal decomposition strategy where the computation for each row of the A matrix
is treated as an independent 1x3 * 3x9 multiplication.
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank for 1x3 * 3x9 is 3.
    # Total rank is n * 3 = 2 * 3 = 6.
    rank_per_block = m
    total_rank = n * rank_per_block
    
    # Initialize factor matrices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank) - Note: Index convention W[k*n + i, r]
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Iterate over each row i of A
    for i in range(n):
        # Offset for this row's rank components
        base_col = i * rank_per_block
        
        for j in range(m):
            for r in range(rank_per_block):
                col = base_col + r
                
                # U index: i * m + j
                # We want U[i, j, col] to be delta(j, r)
                # So U[i*m + j][col] = 1 if j == r else 0
                if j == r:
                    U[i * m + j][col] = 1
                
                # V index: j * p + k
                # We want V[j, k, col] to be delta(j, r) for all k
                # So V[j*p + k][col] = 1 if j == r else 0
                # We need to set this for all k
                for k in range(p):
                    if j == r:
                        V[j * p + k][col] = 1
                
                # W index: k * n + i
                # We want W[k, i, col] to be 1 for all k
                # So W[k*n + i][col] = 1
                for k in range(p):
                    W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
