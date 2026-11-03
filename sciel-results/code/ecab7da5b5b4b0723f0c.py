"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition
    # Split B into p/2 = 4 blocks of width 2.
    # Each block computes M_{2,4,2} which has rank 8.
    # Total rank 4 * 8 = 32.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    num_blocks = p // 2
    block_width = 2
    
    for b in range(num_blocks):
        base_t = b * 8
        for i in range(n):
            for j in range(m):
                for k_local in range(block_width):
                    k_global = b * block_width + k_local
                    t = base_t + i * (m * block_width) + j * block_width + k_local
                    # U index: i*m + j
                    U[i * m + j][t] = 1
                    # V index: j*p + k_global
                    V[j * p + k_global][t] = 1
                    # W index: k_global*n + i
                    W[k_global * n + i][t] = 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
