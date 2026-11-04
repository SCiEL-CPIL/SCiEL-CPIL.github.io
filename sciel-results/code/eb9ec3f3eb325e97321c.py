"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Rank 88 decomposition using 8 blocks of 3x2x2 (rank 11 each).
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for j1 in range(2):
        for k1 in range(4):
            # Strassen part for rows 0,1
            # M1
            U[0 * m + (2 * j1)][col] += 1
            U[1 * m + (2 * j1 + 1)][col] += 1
            V[(2 * j1) * p + (2 * k1)][col] += 1
            V[(2 * j1 + 1) * p + (2 * k1 + 1)][col] += 1
            W[(2 * k1) * n + 0][col] += 1
            W[(2 * k1 + 1) * n + 1][col] += 1
            col += 1
            
            # M2
            U[1 * m + (2 * j1)][col] += 1
            U[1 * m + (2 * j1 + 1)][col] += 1
            V[(2 * j1) * p + (2 * k1)][col] += 1
            W[(2 * k1) * n + 1][col] += 1
            W[(2 * k1 + 1) * n + 1][col] -= 1
            col += 1
            
            # M3
            U[0 * m + (2 * j1)][col] += 1
            V[(2 * j1) * p + (2 * k1 + 1)][col] += 1
            V[(2 * j1 + 1) * p + (2 * k1 + 1)][col] -= 1
            W[(2 * k1 + 1) * n + 0][col] += 1
            W[(2 * k1 + 1) * n + 1][col] += 1
            col += 1
            
            # M4
            U[1 * m + (2 * j1 + 1)][col] += 1
            V[(2 * j1 + 1) * p + (2 * k1)][col] += 1
            V[(2 * j1) * p + (2 * k1)][col] -= 1
            W[(2 * k1) * n + 0][col] += 1
            W[(2 * k1) * n + 1][col] += 1
            col += 1
            
            # M5
            U[0 * m + (2 * j1)][col] += 1
            U[0 * m + (2 * j1 + 1)][col] += 1
            V[(2 * j1 + 1) * p + (2 * k1 + 1)][col] += 1
            W[(2 * k1) * n + 0][col] -= 1
            W[(2 * k1 + 1) * n + 0][col] += 1
            col += 1
            
            # M6
            U[0 * m + (2 * j1 + 1)][col] += 1
            U[1 * m + (2 * j1)][col] -= 1
            V[(2 * j1) * p + (2 * k1)][col] += 1
            V[(2 * j1) * p + (2 * k1 + 1)][col] += 1
            W[(2 * k1 + 1) * n + 1][col] += 1
            col += 1
            
            # M7
            U[1 * m + (2 * j1)][col] += 1
            U[0 * m + (2 * j1)][col] -= 1
            V[(2 * j1 + 1) * p + (2 * k1 + 1)][col] += 1
            V[(2 * j1) * p + (2 * k1)][col] += 1
            W[(2 * k1) * n + 0][col] += 1
            W[(2 * k1) * n + 1][col] += 1
            col += 1
            
            # Linear part for row 2
            # T1
            U[2 * m + (2 * j1)][col] += 1
            V[(2 * j1) * p + (2 * k1)][col] += 1
            W[(2 * k1) * n + 2][col] += 1
            col += 1
            
            # T2
            U[2 * m + (2 * j1 + 1)][col] += 1
            V[(2 * j1 + 1) * p + (2 * k1)][col] += 1
            W[(2 * k1) * n + 2][col] += 1
            col += 1
            
            # T3
            U[2 * m + (2 * j1)][col] += 1
            V[(2 * j1) * p + (2 * k1 + 1)][col] += 1
            W[(2 * k1 + 1) * n + 2][col] += 1
            col += 1
            
            # T4
            U[2 * m + (2 * j1 + 1)][col] += 1
            V[(2 * j1 + 1) * p + (2 * k1 + 1)][col] += 1
            W[(2 * k1 + 1) * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
