"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition of rank 48: 
    # 3 blocks of 4x4x1 effectively? No.
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1).
    # C1 = A B1 (rank 16)
    # C2 = A B2 (rank 16)
    # C3 = A B3 (rank 16)
    # Total rank 48.
    
    n, m, p = DIMENSIONS
    rank = 48
    U = [[0] * rank for _ in range(n * m)] # 16 x 48
    V = [[0] * rank for _ in range(m * p)] # 36 x 48
    W = [[0] * rank for _ in range(p * n)] # 36 x 48
    
    col = 0
    # Block 1: Columns 0-3 of B
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: Columns 4-7 of B
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + (k + 4)][col] = 1
                W[(k + 4) * n + i][col] = 1
                col += 1
                
    # Block 3: Column 8 of B
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
