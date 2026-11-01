"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use two 3x4 blocks, each with 2 Strassen (for 2x2 x 2x4) + 4 rank-1 (for 2x1 x 1x4)
    # Per 3x4 block: 7 (Strassen for cols 0,1) + 7 (Strassen for cols 2,3) + 4 (row 2) = 18
    # But we can share: use 7 for the full 2x2 x 2x4 by doing 2 independent 2x2x2x2
    # Actually, let's try rank 30: use a more efficient global decomposition
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Split B into two 3x4 blocks: B_left (cols 0-3), B_right (cols 4-7)
    # For each 3x4 block, use:
    #   - 7 Strassen terms for A[0:2,0:2] x B[0:2, 0:4] (treating as two 2x2x2x2)
    #   - 4 terms for A[0:2,2] x B[2, 0:4]
    # Total per block: 7 + 7 + 4 = 18, but we can merge the two Strassen parts
    # Actually, 2x2 x 2x4 = two independent 2x2 x 2x2, so 7+7=14 Strassen terms
    # Plus 4 for the third row: 14 + 4 = 18 per 3x4 block
    # Two blocks: 36. Still not better than 44.
    
    # Let me try a different approach: use the fact that we have 3 rows in B
    # and try to find a rank-10 decomposition for 2x3 x 3x4
    # Then double for the two 3x4 blocks: 20 total
    
    # For 2x3 x 3x4, schoolbook is 24. Let's try to get rank 10.
    # This is complex, so let's keep the current 44 and just verify correctness.
    
    # Actually, let's just return the current working solution with rank 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Iterate over the 4 blocks of B (3x2 each)
    for b in range(4):
        # k indices for this block: 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part for A1 (2x2) * B1 (2x2) ---
        # A1 uses A[0,0], A[0,1], A[1,0], A[1,1]
        # B1 uses B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1 # B0k0
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += 1, C1k1 += 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + k0][col] = 1 # B0k0
        # W contributions: C1k0 += 1, C1k1 += -1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col] = 1  # A00
        V[0 * p + k1][col] = 1 # B0k1
        V[1 * p + k1][col] = -1 # B1k1
        # W contributions: C0k1 += 1, C1k1 += 1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col] = 1  # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[0 * p + k0][col] = -1 # B0k0
        # W contributions: C0k0 += 1, C1k0 += 1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += -1, C0k1 += 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col] = 1  # A10
        U[0 * m + 0][col] = -1 # A00
        V[0 * p + k0][col] = 1 # B0k0
        V[0 * p + k1][col] = 1 # B0k1
        # W contributions: C1k1 += 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col] = 1  # A01
        U[1 * m + 1][col] = -1 # A11
        V[1 * p + k0][col] = 1 # B1k0
        V[1 * p + k1][col] = 1 # B1k1
        # W contributions: C0k0 += 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # --- Rank-4 part for A2 (2x1) * B2 (1x2) ---
        # A2 uses A[0,2], A[1,2]
        # B2 uses B[2,k0], B[2,k1]
        
        # P1 = A02 * B2k0
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k0][col] = 1 # B2k0
        # W contributions: C0k0 += 1
        W[k0 * n + 0][col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C0k1 += 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k0][col] = 1 # B2k0
        # W contributions: C1k0 += 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + k1][col] = 1 # B2k1
        # W contributions: C1k1 += 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
