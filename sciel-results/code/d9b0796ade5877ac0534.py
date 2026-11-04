"""Matrix multiplication 2,3,8 with Strassen-based rank reduction."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    # Split 8 columns into 4 blocks of 2
    # Each block: 2x3 x 3x2
    # Use Strassen-like decomposition: rank 11 per block
    # Total rank: 44 -> but let's aim for lower
    
    # Actually, let me use the known result: 2x3x2 has rank at most 11
    # But let me just implement a correct rank-24 solution first, then optimize
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 16 x 24
    
    for block in range(4):
        base = block * 6
        b0 = 2 * block
        b1 = 2 * block + 1
        
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    col = base + i * m * 2 + j * 2 + k
                    U[i * m + j][col] = 1
                    V[j * p + b0 + k][col] = 1
                    W[(b0 + k) * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
