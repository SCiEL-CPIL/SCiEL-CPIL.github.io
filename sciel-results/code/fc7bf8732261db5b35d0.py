"""2x4 by 4x6 matrix multiplication, exact rank-42 tensor decomposition.

The 2x4 matrix A and 4x6 matrix B are partitioned into 2-column and 2-row
blocks, and each pair of blocks is multiplied via the 7-term Strassen scheme.
A has 2 column-blocks (j=0..3 split as 0..1, 2..3).
B has 3 column-blocks (k=0..5 split as 0..1, 2..3, 4..5).
Output C is 2x6.

The problem decomposes into 6 independent 2x2 matrix multiplications:
1. A[:, 0:2] * B[0:2, 0:2]
2. A[:, 2:4] * B[2:4, 0:2]
3. A[:, 0:2] * B[0:2, 2:4]
4. A[:, 2:4] * B[2:4, 2:4]
5. A[:, 0:2] * B[0:2, 4:6]
6. A[:, 2:4] * B[2:4, 4:6]

Each 2x2 multiplication has rank 7 (Strassen).
Total rank = 6 * 7 = 42.
"""

def construct():
    n, m, p = 2, 4, 6
    rank = 42

    # Allocate factor matrices
    # U: (n*m, rank) = (8, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (12, 42)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 block multiply
    # A (2x2), B (2x2), C (2x2)
    # U_s[i, j] = coeff for A[i, j]
    # V_s[j, k] = coeff for B[j, k]
    # W_s[k, i] = coeff for C[i, k] (Note: W index is k*n + i)
    
    # Standard Strassen:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    # U_tbl[s]: list of (i_loc, j_loc, coeff)
    U_tbl = [
        [(0, 0, 1), (1, 1, 1)],         # M0
        [(1, 0, 1), (1, 1, 1)],         # M1
        [(0, 0, 1)],                     # M2
        [(1, 1, 1)],                     # M3
        [(0, 0, 1), (0, 1, 1)],         # M4
        [(1, 0, 1), (0, 0, -1)],        # M5
        [(0, 1, 1), (1, 1, -1)]         # M6
    ]
    
    # V_tbl[s]: list of (j_loc, k_loc, coeff)
    V_tbl = [
        [(0, 0, 1), (1, 1, 1)],         # M0
        [(0, 0, 1)],                     # M1
        [(0, 1, 1), (1, 1, -1)],        # M2
        [(1, 0, 1), (0, 0, -1)],        # M3
        [(1, 1, 1)],                     # M4
        [(0, 0, 1), (0, 1, 1)],         # M5
        [(1, 0, 1), (1, 1, 1)]          # M6
    ]
    
    # W_tbl[s]: list of (i_loc, k_loc, coeff) for C[i, k]
    # Note: The task says W[k*n+i, r]. So row index is k*n + i.
    # My W matrix is indexed by row = k*n + i.
    # So for C[i, k], I add to W[k*n + i].
    W_tbl = [
        [(0, 0, 1), (1, 1, 1)],         # M0: C00 += 1, C11 += 1
        [(1, 0, 1), (1, 1, -1)],        # M1: C10 += 1, C11 -= 1
        [(0, 1, 1), (1, 1, 1)],         # M2: C01 += 1, C11 += 1
        [(0, 0, 1), (1, 0, 1)],         # M3: C00 += 1, C10 += 1
        [(0, 0, -1), (0, 1, 1)],        # M4: C00 -= 1, C01 += 1
        [(1, 1, 1)],                     # M5: C11 += 1
        [(0, 0, 1)]                      # M6: C00 += 1
    ]

    # We have 2 A-blocks (col 0:2, 2:4) and 3 B-blocks (col 0:2, 2:4, 4:6)
    # But B is also split into rows 0:2 and 2:4.
    # So for each B column block, we have 2 row blocks.
    # Total 2 * 3 * 2 = 12 pairs?
    # Wait.
    # A is 2x4. Split into A0 (2x2) and A1 (2x2).
    # B is 4x6.
    # C = A0 * B_top + A1 * B_bot.
    # B_top is 2x6. Split into B_top0 (2x2), B_top1 (2x2), B_top2 (2x2).
    # B_bot is 2x6. Split into B_bot0 (2x2), B_bot1 (2x2), B_bot2 (2x2).
    
    # So we have:
    # 1. A0 * B_top0
    # 2. A1 * B_bot0
    # 3. A0 * B_top1
    # 4. A1 * B_bot1
    # 5. A0 * B_top2
    # 6. A1 * B_bot2
    
    # Indices:
    # A0: rows 0-1, cols 0-1. (j_off=0)
    # A1: rows 0-1, cols 2-3. (j_off=2)
    
    # B_top0: rows 0-1, cols 0-1. (row_off=0, k_off=0)
    # B_bot0: rows 2-3, cols 0-1. (row_off=2, k_off=0)
    # B_top1: rows 0-1, cols 2-3. (row_off=0, k_off=2)
    # B_bot1: rows 2-3, cols 2-3. (row_off=2, k_off=2)
    # B_top2: rows 0-1, cols 4-5. (row_off=0, k_off=4)
    # B_bot2: rows 2-3, cols 4-5. (row_off=2, k_off=4)

    pairs = [
        (0, 0, 0),  # A0, B_top0
        (1, 2, 0),  # A1, B_bot0
        (0, 0, 2),  # A0, B_top1
        (1, 2, 2),  # A1, B_bot1
        (0, 0, 4),  # A0, B_top2
        (1, 2, 4)   # A1, B_bot2
    ]

    for idx, (a_idx, b_row_off, b_col_off) in enumerate(pairs):
        a_col_off = a_idx * 2
        base_col = idx * 7

        for s in range(7):
            col = base_col + s
            
            # Fill U
            for i_loc, j_loc, c in U_tbl[s]:
                U[i_loc * m + (a_col_off + j_loc)][col] += c

            # Fill V
            for j_loc, k_loc, c in V_tbl[s]:
                V[(b_row_off + j_loc) * p + (b_col_off + k_loc)][col] += c

            # Fill W
            for i_loc, k_loc, c in W_tbl[s]:
                W[(b_col_off + k_loc) * n + i_loc][col] += c

    return {"U": U, "V": V, "W": W}
