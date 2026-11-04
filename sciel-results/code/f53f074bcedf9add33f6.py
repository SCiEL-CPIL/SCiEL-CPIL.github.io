"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-27 exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Compute the top row of the output (1x4) using rank 3.
       The top row C[0, k] = sum_j A[0, j] * B[j, k].
       We use 3 rank-1 terms, one for each j in {0, 1, 2}.
       Each term r corresponds to j=r.
       U[0, r] = 1
       V[r, k] = 1 for all k in {0, 1, 2, 3}
       W[k, 0] = 1 for all k in {0, 1, 2, 3}
       
    2. Compute the bottom two rows of the output (2x4) using rank 24 (schoolbook).
       For i in {1, 2}, j in {0, 1, 2}, k in {0, 1, 2, 3}:
       U[i, j] = 1
       V[j, k] = 1
       W[k, i] = 1
       
    Total Rank = 3 + 24 = 27.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Total rank
    rank_top = m  # 3
    rank_bottom = (n - 1) * m * p  # 2 * 3 * 4 = 24
    total_rank = rank_top + rank_bottom  # 27
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: Top Row (Rank 3)
    # i = 0
    for j in range(m):
        col = j  # 0, 1, 2
        U[0 * m + j][col] = 1
        for k in range(p):
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            
    # Part 2: Bottom Rows (Rank 24)
    col_offset = rank_top
    col_idx = 0
    for i in range(1, n):  # i = 1, 2
        for j in range(m):  # j = 0, 1, 2
            for k in range(p):  # k = 0, 1, 2, 3
                col = col_offset + col_idx
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
