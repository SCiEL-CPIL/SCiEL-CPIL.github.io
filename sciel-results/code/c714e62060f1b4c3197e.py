"""Rewritten program for 2x3 * 3x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    # We split the 3x8 matrix B into four 3x2 blocks.
    # For each block, we decompose A (2x3) * B_block (3x2) into:
    #   - A_top (2x2) * B_top (2x2): rank 7 (Strassen)
    #   - A_bottom (2x1) * B_bottom (1x2): rank 4 (trivial)
    # Total per block: 11, total rank: 4 * 11 = 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for A_top (rows 0,1 of A; cols 0,1) * B_top (rows 0,1 of B; cols k0,k1)
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1; U[1][col] = 1
        V[k0][col] = 1; V[p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3][col] = 1; U[1][col] = 1
        V[k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0][col] = 1
        V[p + k1][col] = 1; V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1][col] = 1
        V[p + k0][col] = 1; V[k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0][col] = 1; U[1][col] = 1
        V[p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[p + k0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1; U[4][col] = -1
        V[p + k0][col] = 1; V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # Rank-4 part for A_bottom (col 2 of A) * B_bottom (row 2 of B; cols k0,k1)
        # P1 = A02 * B2k0
        U[2][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[5][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[5][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
