"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into 3 blocks of 4x3
    # Each 4x3 block has rank 12 (schoolbook)
    # Total rank 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for block in range(3):
        # k_offset is the starting column index for this block in B and C
        # B is 4x9, so block 0 uses cols 1-3, block 1 uses 4-6, block 2 uses 7-9
        # In 0-indexed: 0-2, 3-5, 6-8
        k_offset = block * 3
        
        for j in range(m):
            for k_local in range(3):
                k_global = k_offset + k_local
                
                # U: depends on i, j. For fixed j, all i have 1.
                for i in range(n):
                    U[i * m + j][col] = 1
                
                # V: depends on j, k. Fixed j, k_global.
                V[j * p + k_global][col] = 1
                
                # W: depends on k, i. For fixed k_global, all i have 1.
                for i in range(n):
                    W[k_global * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
