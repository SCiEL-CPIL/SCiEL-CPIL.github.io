"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Split p into 5 blocks of 2
    num_blocks = p // 2
    # For each block, rank 3 for I_{2x2}
    # For each j in 0..2, we use these 3 ranks
    # Total rank = 3 * 5 * 3 = 45
    rank = 45
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for t in range(num_blocks):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # Term 1: i=0, k=k0 -> i'=0, k'=k0
            U[0 * m + j][col] += 1
            V[j * p + k0][col] += 1
            W[k0 * n + 0][col] += 1
            col += 1
            
            # Term 2: i=0, k=k1 -> i'=0, k'=k1
            U[0 * m + j][col] += 1
            V[j * p + k1][col] += 1
            W[k1 * n + 0][col] += 1
            col += 1
            
            # Term 3: i=1, k=k0, k1 -> i'=1, k'=k0, k1
            U[1 * m + j][col] += 1
            V[j * p + k0][col] += 1
            V[j * p + k1][col] += 1
            W[k0 * n + 1][col] += 1
            W[k1 * n + 1][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
