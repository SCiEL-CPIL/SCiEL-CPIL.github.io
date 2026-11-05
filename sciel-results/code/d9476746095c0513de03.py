"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal-ish exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Use a 2-level recursive Strassen decomposition.
    Level 1: Split 6x6 into 2x2 blocks of 3x3. Use Strassen (7 multiplications of 3x3).
    Level 2: For each 3x3 multiplication, use a known rank-23 decomposition (Bayer's method
             or equivalent). Actually, let me use a simpler approach:
             
    Since 3x3 multiplication has known rank <= 23 (and the best known is 23),
    we can use that. But constructing it explicitly is complex.
    
    Alternative: Use 2-level Strassen where level 2 splits 3x3 into blocks of sizes
    that allow further Strassen. But 3 is odd, so we can't do clean 2x2 splitting.
    
    Better approach: Pad 3x3 to 4x4 and use Strassen on 4x4 (rank 28), then
    total rank = 7 * 28 = 196. That's what the original was trying.
    
    Even better: Use the fact that R(6,6,6) <= 7 * R(3,3,3) and R(3,3,3) = 23
    is known. So rank 161 should be achievable. But constructing the rank-23
    decomposition for 3x3 is non-trivial.
    
    Let me use a different strategy: Direct 2-level Strassen where we split
    6x6 into blocks of sizes (3,3) at level 1, and for each 3x3 product,
    we use schoolbook (rank 27). Total: 7 * 27 = 189.
    
    This is what the current program does. Let me verify it's correct and
    try to improve by using a better 3x3 decomposition.
    
    Actually, let me try a completely different approach: use the Winograd
    algorithm or direct construction of a lower-rank decomposition.
    
    For now, let me stick with the 189-rank approach but make sure it's
    correctly implemented.
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Strassen terms at level 1 for 6x6 = 2x2 blocks of 3x3
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # Each block is 3x3
    #
    # T1: (A00+A11)*(B00+B11)
    # T2: (A10+A11)*B00
    # T3: A00*(B01-B11)
    # T4: A11*(B10-B00)
    # T5: (A00+A01)*B11
    # T6: (A10-A00)*(B00+B01)
    # T7: (A01-A11)*(B10+B11)
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]
    
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6
    result_contribs = {
        (0,0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0,1): [(2, 1), (4, 1)],
        (1,0): [(1, 1), (3, 1)],
        (1,1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    col = 0
    for t_idx, (left_combo, right_combo) in enumerate(strassen_terms):
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U: coefficients of A entries
                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l
                    
                    # V: coefficients of B entries
                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r
                    
                    # W: contribution to C[i_total, k_total]
                    for (rbr, rbc), signs in result_contribs.items():
                        for (si, sign) in signs:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                W_row = k_total * 6 + i_total
                                W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
