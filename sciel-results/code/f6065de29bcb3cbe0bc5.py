"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split A into A_left (2x2) and A_right (2x1)
    # Split B into B_top (2x9) and B_bottom (1x9)
    # C = A_left B_top + A_right B_bottom
    
    # Part 1: A_left (2x2) * B_top (2x9)
    # B_top is split into 3 blocks of (2x3)
    # Each (2x2)*(2x3) uses schoolbook rank 6
    # Total rank for Part 1: 3 * 6 = 18
    
    # Part 2: A_right (2x1) * B_bottom (1x9)
    # This is outer product of 2x1 and 1x9
    # Rank is 9 (one for each column)
    # Total rank for Part 2: 9
    
    # Total Rank = 18 + 9 = 27
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)] # 6x27
    V = [[0] * rank for _ in range(m * p)] # 27x27
    W = [[0] * rank for _ in range(p * n)] # 18x27
    
    col = 0
    
    # --- Part 1: A_left (rows 0,1; cols 0,1 of A) * B_top (rows 0,1 of B) ---
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..8
    # Split k into 3 blocks of 3: k0 in 0..2, k1 in 3..5, k2 in 6..8
    
    for block in range(3):
        k_offset = block * 3
        for i in range(2):
            for j in range(2):
                for k_local in range(3):
                    k = k_offset + k_local
                    # U[i*m + j] -> U[i*3 + j]
                    U[i * 3 + j][col] = 1
                    # V[j*p + k] -> V[j*9 + k]
                    V[j * 9 + k][col] = 1
                    # W[k*n + i] -> W[k*2 + i]
                    W[k * 2 + i][col] = 1
                    col += 1
                    
    # --- Part 2: A_right (row 0,1; col 2 of A) * B_bottom (row 2 of B) ---
    # A indices: i in 0..1, j = 2
    # B indices: j = 2, k in 0..8
    # C[i,k] += A[i,2] * B[2,k]
    
    for i in range(2):
        for k in range(9):
            U[i * 3 + 2][col] = 1
            V[2 * 9 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
