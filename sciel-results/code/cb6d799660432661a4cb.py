"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of M_{2,3,8} using Strassen-style optimization for the 2x2 block.
    # A is 2x3: A = [A_L (2x2) | A_R (2x1)]
    # B is 3x8: B = [B_T (2x8); B_B (1x8)]
    # C = A_L @ B_T + A_R @ B_B
    #
    # For A_L @ B_T (2x2 x 2x8), we use 4 Strassen-like multiplications (one per column of B_T).
    # This reduces the 2x2x8 multiplication from 32 scalar mults to 4 * 7 = 28.
    # For A_R @ B_B (2x1 x 1x8), we use standard multiplication: 2*1*8 = 16.
    # Total rank = 28 + 16 = 44.
    #
    # U: (n*m, rank) = (6, 44)
    # V: (m*p, rank) = (24, 44)
    # W: (p*n, rank) = (16, 44)
    #
    # Indexing:
    # U row index: i * m + j, where i in [0,1], j in [0,2]
    # V row index: j * p + k, where j in [0,2], k in [0,7]
    # W row index: k * n + i, where k in [0,7], i in [0,1]  (trace convention W[k*n+i, r])
    
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Strassen-like decomposition for A_L (2x2) @ B_T (2x8)
    # A_L rows: A_00, A_01, A_10, A_11 -> U indices 0, 1, 3, 4 (since j=0,1 for left block)
    #   A_00: U[0*3+0] = U[0]
    #   A_01: U[0*3+1] = U[1]
    #   A_10: U[1*3+0] = U[3]
    #   A_11: U[1*3+1] = U[4]
    # B_T rows: B_00..B_07 (row j=0), B_10..B_17 (row j=1) -> V indices
    #   B_j,k: V[j*8+k]
    #   B_00: V[0], B_01: V[1], ..., B_07: V[7]
    #   B_10: V[8], B_11: V[9], ..., B_17: V[15]
    # C_L is 2x8: C_ik for i in [0,1], k in [0,7] -> W indices k*n+i = 2*k+i
    #   C_00: W[0], C_01: W[2], ..., C_07: W[14]
    #   C_10: W[1], C_11: W[3], ..., C_17: W[15]
    
    # Strassen for 2x2 @ 2x1 (per column k):
    # M1 = A_00 * (B_0k + B_1k)
    # M2 = (A_00 + A_01) * B_1k
    # M3 = A_01 * (B_0k - B_1k)
    # M4 = A_10 * (B_0k + B_1k)
    # M5 = (A_10 + A_11) * B_0k
    # M6 = A_11 * (B_0k - B_1k)
    # M7 = (A_00 + A_11) * (B_1k - B_0k)
    # 
    # C_0k = M1 + M3 - M6 + M7
    # C_1k = M4 - M2 + M5 + M6
    
    strassen_terms = [
        # (U_coeffs: dict {row: val}, V_coeffs: dict {row: val}, W_coeffs: dict {row: val})
        # M1: A_00 * (B_0k + B_1k)
        ({0: 1}, {k: 1 for k in range(8)}, {}),  # Placeholder, will be per-column
    ]
    
    # We need to handle this per-column since V and W depend on k.
    # Let's iterate over columns k=0..7 and add 7 terms each.
    
    for k in range(8):
        # M1: A_00 * (B_0k + B_1k)
        # U: A_00 -> U[0] = 1
        # V: B_0k -> V[0*8+k] = 1, B_1k -> V[1*8+k] = 1
        # W: C_0k -> W[2*k+0] = 1, C_1k -> W[2*k+1] = 1
        U[0][col] = 1
        V[k][col] = 1
        V[8+k][col] = 1
        W[2*k][col] = 1
        W[2*k+1][col] = 1
        col += 1
        
        # M2: (A_00 + A_01) * B_1k
        # U: A_00 -> U[0] = 1, A_01 -> U[1] = 1
        # V: B_1k -> V[8+k] = 1
        # W: C_1k -> W[2*k+1] = -1  (C_1k = M4 - M2 + M5 + M6)
        U[0][col] = 1
        U[1][col] = 1
        V[8+k][col] = 1
        W[2*k+1][col] = -1
        col += 1
        
        # M3: A_01 * (B_0k - B_1k)
        # U: A_01 -> U[1] = 1
        # V: B_0k -> V[k] = 1, B_1k -> V[8+k] = -1
        # W: C_0k -> W[2*k] = 1
        U[1][col] = 1
        V[k][col] = 1
        V[8+k][col] = -1
        W[2*k][col] = 1
        col += 1
        
        # M4: A_10 * (B_0k + B_1k)
        # U: A_10 -> U[3] = 1
        # V: B_0k -> V[k] = 1, B_1k -> V[8+k] = 1
        # W: C_1k -> W[2*k+1] = 1
        U[3][col] = 1
        V[k][col] = 1
        V[8+k][col] = 1
        W[2*k+1][col] = 1
        col += 1
        
        # M5: (A_10 + A_11) * B_0k
        # U: A_10 -> U[3] = 1, A_11 -> U[4] = 1
        # V: B_0k -> V[k] = 1
        # W: C_1k -> W[2*k+1] = 1
        U[3][col] = 1
        U[4][col] = 1
        V[k][col] = 1
        W[2*k+1][col] = 1
        col += 1
        
        # M6: A_11 * (B_0k - B_1k)
        # U: A_11 -> U[4] = 1
        # V: B_0k -> V[k] = 1, B_1k -> V[8+k] = -1
        # W: C_0k -> W[2*k] = -1, C_1k -> W[2*k+1] = 1
        U[4][col] = 1
        V[k][col] = 1
        V[8+k][col] = -1
        W[2*k][col] = -1
        W[2*k+1][col] = 1
        col += 1
        
        # M7: (A_00 + A_11) * (B_1k - B_0k)
        # U: A_00 -> U[0] = 1, A_11 -> U[4] = 1
        # V: B_0k -> V[k] = -1, B_1k -> V[8+k] = 1
        # W: C_0k -> W[2*k] = 1
        U[0][col] = 1
        U[4][col] = 1
        V[k][col] = -1
        V[8+k][col] = 1
        W[2*k][col] = 1
        col += 1
    
    # Part 2: Standard multiplication for A_R (2x1) @ B_B (1x8)
    # A_R: A_02 -> U[2], A_12 -> U[5]
    # B_B: B_2k -> V[16+k]
    # C: C_ik -> W[2*k+i]
    # Term: A_i2 * B_2k * C_ik
    for k in range(8):
        # i=0
        U[2][col] = 1
        V[16+k][col] = 1
        W[2*k][col] = 1
        col += 1
        # i=1
        U[5][col] = 1
        V[16+k][col] = 1
        W[2*k+1][col] = 1
        col += 1
    
    # col should be 44
    assert col == 44, f"Expected 44, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
