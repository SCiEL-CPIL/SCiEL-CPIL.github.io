"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Two 3x4 blocks, each decomposed into two 3x2 Strassen blocks.
    # Rank per 3x2 block is 11. Total rank 4 * 11 = 44.
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 3x2 block decomposition starting at column k_start
    def add_3x2_block(k_start, col_offset):
        nonlocal col
        k0 = k_start
        k1 = k_start + 1
        
        # --- Strassen part for A1 (2x2) * B1 (2x2) ---
        # A1 uses A[0,0], A[0,1], A[1,0], A[1,1]
        # B1 uses B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * m + 0][col_offset] = 1
        U[1 * m + 1][col_offset] = 1
        V[0 * p + k0][col_offset] = 1
        V[1 * p + k1][col_offset] = 1
        W[k0 * n + 0][col_offset] = 1
        W[k1 * n + 1][col_offset] = 1
        col_offset += 1
        
        # M2 = (A10 + A11) * B0k0
        U[1 * m + 0][col_offset] = 1
        U[1 * m + 1][col_offset] = 1
        V[0 * p + k0][col_offset] = 1
        W[k0 * n + 1][col_offset] = 1
        W[k1 * n + 1][col_offset] = -1
        col_offset += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0 * m + 0][col_offset] = 1
        V[0 * p + k1][col_offset] = 1
        V[1 * p + k1][col_offset] = -1
        W[k1 * n + 0][col_offset] = 1
        W[k1 * n + 1][col_offset] = 1
        col_offset += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[1 * m + 1][col_offset] = 1
        V[1 * p + k0][col_offset] = 1
        V[0 * p + k0][col_offset] = -1
        W[k0 * n + 0][col_offset] = 1
        W[k0 * n + 1][col_offset] = 1
        col_offset += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0 * m + 0][col_offset] = 1
        U[0 * m + 1][col_offset] = 1
        V[1 * p + k1][col_offset] = 1
        W[k0 * n + 0][col_offset] = -1
        W[k1 * n + 0][col_offset] = 1
        col_offset += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * m + 0][col_offset] = 1
        U[0 * m + 0][col_offset] = -1
        V[0 * p + k0][col_offset] = 1
        V[0 * p + k1][col_offset] = 1
        W[k1 * n + 1][col_offset] = 1
        col_offset += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * m + 1][col_offset] = 1
        U[1 * m + 1][col_offset] = -1
        V[1 * p + k0][col_offset] = 1
        V[1 * p + k1][col_offset] = 1
        W[k0 * n + 0][col_offset] = 1
        col_offset += 1
        
        # --- Rank-4 part for A2 (2x1) * B2 (1x2) ---
        # A2 uses A[0,2], A[1,2]
        # B2 uses B[2,k0], B[2,k1]
        
        # P1 = A02 * B2k0
        U[0 * m + 2][col_offset] = 1
        V[2 * p + k0][col_offset] = 1
        W[k0 * n + 0][col_offset] = 1
        col_offset += 1
        
        # P2 = A02 * B2k1
        U[0 * m + 2][col_offset] = 1
        V[2 * p + k1][col_offset] = 1
        W[k1 * n + 0][col_offset] = 1
        col_offset += 1
        
        # P3 = A12 * B2k0
        U[1 * m + 2][col_offset] = 1
        V[2 * p + k0][col_offset] = 1
        W[k0 * n + 1][col_offset] = 1
        col_offset += 1
        
        # P4 = A12 * B2k1
        U[1 * m + 2][col_offset] = 1
        V[2 * p + k1][col_offset] = 1
        W[k1 * n + 1][col_offset] = 1
        col_offset += 1
        
        return col_offset

    # Block 1: Columns 0-3
    col = add_3x2_block(0, 0)
    col = add_3x2_block(2, col)
    
    # Block 2: Columns 4-7
    col = add_3x2_block(4, col)
    col = add_3x2_block(6, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
