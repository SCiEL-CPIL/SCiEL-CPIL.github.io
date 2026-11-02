"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    # Decompose p=9 as 3 blocks of size 3
    num_blocks = p // 3  # 3
    block_size = 3       # 3
    
    # Rank for each 2x3x3 block is 6 (schoolbook for 1x3x3 is 3, times 2 rows)
    rank_per_block = n * block_size  # 2 * 3 = 6
    rank = num_blocks * rank_per_block  # 3 * 6 = 18
    
    U = [[0] * rank for _ in range(n * m)]   # (6, 18)
    V = [[0] * rank for _ in range(m * p)]   # (27, 18)
    W = [[0] * rank for _ in range(p * n)]   # (18, 18)
    
    col = 0
    for t in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(block_size):
                    # U index: i*m + j
                    U[i * m + j][col] = 1
                    # V index: j*p + (t*block_size + k)
                    V[j * p + t * block_size + k][col] = 1
                    # W index: (t*block_size + k)*n + i
                    W[(t * block_size + k) * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
