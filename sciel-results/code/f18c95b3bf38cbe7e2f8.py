"""Matrix multiplication 3x4x6 - rank 54 decomposition via 2x2x2 and 1x2x2 blocks."""

def construct():
    n, m, p = 3, 4, 6
    
    # Total rank = 3 (B col splits) * (2 * 7 (Strassen for 2x2x2) + 2 * 2 (Simple for 1x2x2))
    # = 3 * (14 + 4) = 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, coeff in u_coeffs:
            U[idx][col] += coeff
        for idx, coeff in v_coeffs:
            V[idx][col] += coeff
        for idx, coeff in w_coeffs:
            W[idx][col] += coeff
        col += 1

    # Loop over 3 column pairs of B (and C)
    for b_col_pair in range(3):
        k0 = 2 * b_col_pair
        k1 = 2 * b_col_pair + 1
        
        # C indices for this column pair
        # C[i, k] index is k * n + i
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        c20 = k0 * n + 2
        c21 = k1 * n + 2
        
        # A indices
        # A11: Rows 0-1, Cols 0-1
        a11_00 = 0 * m + 0
        a11_01 = 0 * m + 1
        a11_10 = 1 * m + 0
        a11_11 = 1 * m + 1
        
        # A12: Rows 0-1, Cols 2-3
        a12_00 = 0 * m + 2
        a12_01 = 0 * m + 3
        a12_10 = 1 * m + 2
        a12_11 = 1 * m + 3
        
        # A21: Row 2, Cols 0-1
        a21_00 = 2 * m + 0
        a21_01 = 2 * m + 1
        
        # A22: Row 2, Cols 2-3
        a22_00 = 2 * m + 2
        a22_01 = 2 * m + 3
        
        # B indices for this column pair
        # B_top: Rows 0-1, Cols k0, k1
        b_top_00 = 0 * p + k0
        b_top_01 = 0 * p + k1
        b_top_10 = 1 * p + k0
        b_top_11 = 1 * p + k1
        
        # B_bot: Rows 2-3, Cols k0, k1
        b_bot_00 = 2 * p + k0
        b_bot_01 = 2 * p + k1
        b_bot_10 = 3 * p + k0
        b_bot_11 = 3 * p + k1
        
        # --- Part 1: A11 * B_top (2x2x2 Strassen) ---
        # M1 = (a11_00 + a11_11)(b_top_00 + b_top_11) -> c00, c11
        add_term(
            [(a11_00, 1), (a11_11, 1)],
            [(b_top_00, 1), (b_top_11, 1)],
            [(c00, 1), (c11, 1)]
        )
        # M2 = (a11_10 + a11_11)b_top_00 -> c10, -c11
        add_term(
            [(a11_10, 1), (a11_11, 1)],
            [(b_top_00, 1)],
            [(c10, 1), (c11, -1)]
        )
        # M3 = a11_00(b_top_01 - b_top_11) -> c01, c11
        add_term(
            [(a11_00, 1)],
            [(b_top_01, 1), (b_top_11, -1)],
            [(c01, 1), (c11, 1)]
        )
        # M4 = a11_11(b_top_10 - b_top_00) -> c00, c10
        add_term(
            [(a11_11, 1)],
            [(b_top_10, 1), (b_top_00, -1)],
            [(c00, 1), (c10, 1)]
        )
        # M5 = (a11_00 + a11_01)b_top_11 -> -c00, c01
        add_term(
            [(a11_00, 1), (a11_01, 1)],
            [(b_top_11, 1)],
            [(c00, -1), (c01, 1)]
        )
        # M6 = (a11_10 - a11_00)(b_top_00 + b_top_01) -> c11
        add_term(
            [(a11_10, 1), (a11_00, -1)],
            [(b_top_00, 1), (b_top_01, 1)],
            [(c11, 1)]
        )
        # M7 = (a11_01 - a11_11)(b_top_10 + b_top_11) -> c00
        add_term(
            [(a11_01, 1), (a11_11, -1)],
            [(b_top_10, 1), (b_top_11, 1)],
            [(c00, 1)]
        )
        
        # --- Part 2: A12 * B_bot (2x2x2 Strassen) ---
        # M1 = (a12_00 + a12_11)(b_bot_00 + b_bot_11) -> c00, c11
        add_term(
            [(a12_00, 1), (a12_11, 1)],
            [(b_bot_00, 1), (b_bot_11, 1)],
            [(c00, 1), (c11, 1)]
        )
        # M2 = (a12_10 + a12_11)b_bot_00 -> c10, -c11
        add_term(
            [(a12_10, 1), (a12_11, 1)],
            [(b_bot_00, 1)],
            [(c10, 1), (c11, -1)]
        )
        # M3 = a12_00(b_bot_01 - b_bot_11) -> c01, c11
        add_term(
            [(a12_00, 1)],
            [(b_bot_01, 1), (b_bot_11, -1)],
            [(c01, 1), (c11, 1)]
        )
        # M4 = a12_11(b_bot_10 - b_bot_00) -> c00, c10
        add_term(
            [(a12_11, 1)],
            [(b_bot_10, 1), (b_bot_00, -1)],
            [(c00, 1), (c10, 1)]
        )
        # M5 = (a12_00 + a12_01)b_bot_11 -> -c00, c01
        add_term(
            [(a12_00, 1), (a12_01, 1)],
            [(b_bot_11, 1)],
            [(c00, -1), (c01, 1)]
        )
        # M6 = (a12_10 - a12_00)(b_bot_00 + b_bot_01) -> c11
        add_term(
            [(a12_10, 1), (a12_00, -1)],
            [(b_bot_00, 1), (b_bot_01, 1)],
            [(c11, 1)]
        )
        # M7 = (a12_01 - a12_11)(b_bot_10 + b_bot_11) -> c00
        add_term(
            [(a12_01, 1), (a12_11, -1)],
            [(b_bot_10, 1), (b_bot_11, 1)],
            [(c00, 1)]
        )
        
        # --- Part 3: A21 * B_top (1x2x2 Simple) ---
        # Term 1: a21_00 * (b_top_00, b_top_01) -> c20, c21
        add_term(
            [(a21_00, 1)],
            [(b_top_00, 1), (b_top_01, 1)],
            [(c20, 1), (c21, 1)]
        )
        # Term 2: a21_01 * (b_top_10, b_top_11) -> c20, c21
        add_term(
            [(a21_01, 1)],
            [(b_top_10, 1), (b_top_11, 1)],
            [(c20, 1), (c21, 1)]
        )
        
        # --- Part 4: A22 * B_bot (1x2x2 Simple) ---
        # Term 1: a22_00 * (b_bot_00, b_bot_01) -> c20, c21
        add_term(
            [(a22_00, 1)],
            [(b_bot_00, 1), (b_bot_01, 1)],
            [(c20, 1), (c21, 1)]
        )
        # Term 2: a22_01 * (b_bot_10, b_bot_11) -> c20, c21
        add_term(
            [(a22_01, 1)],
            [(b_bot_10, 1), (b_bot_11, 1)],
            [(c20, 1), (c21, 1)]
        )

    return {"U": U, "V": V, "W": W}
