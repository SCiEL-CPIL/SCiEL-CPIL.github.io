"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    import numpy as np

    # Strategy: 2x12 = four independent 2x3 blocks
    # Find a verified low-rank decomposition for M_{2,2,3}
    # Then use block-diagonal (Kronecker) structure for total rank = 4 * rank_223

    # Build target tensor for 2x2 by 2x3
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b02, 3=b10, 4=b11, 5=b12
    # C indices: 0=c00, 1=c01, 2=c02, 3=c10, 4=c11, 5=c12
    T = np.zeros((4, 6, 6), dtype=float)
    T[0, 0, 0] = 1; T[1, 3, 0] = 1  # c00
    T[0, 1, 1] = 1; T[1, 4, 1] = 1  # c01
    T[0, 2, 2] = 1; T[1, 5, 2] = 1  # c02
    T[2, 0, 3] = 1; T[3, 3, 3] = 1  # c10
    T[2, 1, 4] = 1; T[3, 4, 4] = 1  # c11
    T[2, 2, 5] = 1; T[3, 5, 5] = 1  # c12

    # Try rank-5 decomposition for M_{2,2,3}
    # Use Strassen-like ideas: combine terms
    # Let's try a systematic search with small integer/half-integer coefficients
    
    # Candidate rank-5 decomposition (verified below):
    # Term structure based on sharing between rows
    A5 = np.array([
        [1, 0, 0, 0],   # a00
        [0, 1, 0, 0],   # a01
        [0, 0, 1, 0],   # a10
        [0, 0, 0, 1],   # a11
        [1, 1, 1, 1],   # a00+a01+a10+a11
    ], dtype=float)
    
    B5 = np.array([
        [1, 0, 0, 0, 0, 0],   # b00
        [0, 1, 0, 0, 0, 0],   # b01
        [0, 0, 1, 0, 0, 0],   # b02
        [0, 0, 0, 1, 0, 0],   # b10
        [0, 0, 0, 0, 1, 1],   # b11+b12
    ], dtype=float)
    
    C5 = np.array([
        [1, 0, 0, 0, 0, 0],   # c00
        [0, 1, 0, 0, 0, 0],   # c01
        [0, 0, 1, 0, 0, 0],   # c02
        [0, 0, 0, 1, 0, 0],   # c10
        [0, 0, 0, 0, 1, 1],   # c11+c12
    ], dtype=float)
    
    # Verify
    T_recon = np.einsum('ri,rj,rk->ijk', A5, B5, C5)
    
    if not np.allclose(T_recon, T):
        # Try another rank-5 candidate
        A5 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, 1, -1],
        ], dtype=float)
        B5 = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ], dtype=float)
        C5 = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ], dtype=float)
        T_recon = np.einsum('ri,rj,rk->ijk', A5, B5, C5)
    
    if not np.allclose(T_recon, T):
        # Use rank-6 (schoolbook) as fallback for 2x2x2x3
        # Schoolbook: 2 rows x 3 cols = 6 terms
        A6 = np.zeros((6, 4))
        B6 = np.zeros((6, 6))
        C6 = np.zeros((6, 6))
        for r in range(6):
            i = r // 3  # row of A
            k = r % 3   # col of C
            A6[r, i] = 1
            B6[r, k] = 1
            C6[r, k] = 1
        # This is not quite right for schoolbook. Let me do it properly.
        # Schoolbook for 2x2 by 2x3:
        # c00 = a00*b00 + a01*b10
        # c01 = a00*b01 + a01*b11
        # c02 = a00*b02 + a01*b12
        # c10 = a10*b00 + a11*b10
        # c11 = a10*b01 + a11*b11
        # c12 = a10*b02 + a11*b12
        # Each c_ij has 2 terms, total 12 terms but we can group by (i,k) pair
        # Actually schoolbook is 6 terms if we do it as: for each output col, 2 mults
        # Let me just use rank 6 with proper terms
        A6 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=float)
        B6 = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float)
        C6 = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float)
        T_recon = np.einsum('ri,rj,rk->ijk', A6, B6, C6)
        if not np.allclose(T_recon, T):
            # Ultimate fallback: full schoolbook for 2x2x2x12
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
        rank_223 = 6
        A_block = A6
        B_block = B6
        C_block = C6
    else:
        rank_223 = 5
        A_block = A5
        B_block = B5
        C_block = C5

    # Now build the full 2x2 by 2x12 decomposition
    # B is 2x12, split into 4 blocks of 2x3: B[:, 0:3], B[:, 3:6], B[:, 6:9], B[:, 9:12]
    # C is 2x12, split into 4 blocks of 2x3: C[:, 0:3], C[:, 3:6], C[:, 6:9], C[:, 9:12]
    # Total rank = 4 * rank_223
    total_rank = 4 * rank_223
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x total_rank

    for block in range(4):
        base = block * rank_223
        for r in range(rank_223):
            col = base + r
            # A_block[r, i] for i in 0..3 -> U[i][col]
            for i in range(4):
                U[i][col] = A_block[r, i]
            # B_block[r, j] for j in 0..5 -> V[block*6 + j][col]
            for j in range(6):
                V[block * 6 + j][col] = B_block[r, j]
            # C_block[r, k] for k in 0..5 -> W[block*6 + k][col]
            # W index: k*n + i where k is column of C (0..11), i is row of C (0..1)
            # For block, column offset = block*3, so C col = block*3 + k_local
            # W index = (block*3 + k_local) * n + i_row
            # But C_block[r, k] where k is 0..5 maps to c00,c01,c02,c10,c11,c12
            # k=0 -> c00: row 0, col 0
            # k=1 -> c01: row 0, col 1
            # k=2 -> c02: row 0, col 2
            # k=3 -> c10: row 1, col 0
            # k=4 -> c11: row 1, col 1
            # k=5 -> c12: row 1, col 2
            for k in range(6):
                c_row = k // 3
                c_col_local = k % 3
                c_col_global = block * 3 + c_col_local
                w_idx = c_col_global * n + c_row
                W[w_idx][col] = C_block[r, k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
