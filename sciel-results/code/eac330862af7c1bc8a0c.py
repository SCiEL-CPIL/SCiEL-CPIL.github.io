"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: One level of Strassen (7 terms) on 3x3 blocks, each 3x3 block
    multiplication decomposed via schoolbook (27 scalar multiplications).
    Total rank = 7 * 27 = 189.
    
    The decomposition follows the trace convention:
      C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where n=m=p=6.
    """
    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Split 6x6 into 2x2 grid of 3x3 blocks:
    #   A = [[A00, A01],   B = [[B00, B01],   C = [[C00, C01],
    #         [A10, A11]]       [B10, B11]]       [C10, C11]]
    # Block (br, bc) covers rows 3*br..3*br+2, cols 3*bc..3*bc+2
    
    # 7 Strassen terms: each computes X * Y where X, Y are 3x3 linear combos
    # of the 3x3 blocks of A and B respectively.
    # Each term is a list of (block_row, block_col, coefficient) for the linear combo.
    
    strassen_terms = [
        # T0: (A00 + A11) * (B00 + B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # T1: (A10 + A11) * B00
        ([(1,0,1), (1,1,1)], [(0,0,1)]),
        # T2: A00 * (B01 - B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),
        # T3: A11 * (B10 - B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),
        # T4: (A00 + A01) * B11
        ([(0,0,1), (0,1,1)], [(1,1,1)]),
        # T5: (A10 - A00) * (B00 + B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # T6: (A01 - A11) * (B10 + B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
    ]
    
    # Result block contributions: C_block = sum of signed Strassen terms
    # C00 = T0 + T3 - T4 + T6
    # C01 = T2 + T4
    # C10 = T1 + T3
    # C11 = T0 - T1 + T2 + T5
    result_contribs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)],
    }
    
    col = 0
    for t_idx in range(7):
        left_combo, right_combo = strassen_terms[t_idx]
        # left_combo: list of (br, bc, coeff) for A blocks
        # right_combo: list of (br, bc, coeff) for B blocks
        
        # Schoolbook decomposition of 3x3 * 3x3:
        # Z[i,k] = sum_a X[i,a] * Y[a,k]
        # For each (i, a, k), one scalar multiplication.
        for i in range(3):
            for a in range(3):
                for k in range(3):
                    # U: X[i,a] = sum over left_combo entries
                    #   X[i,a] = sum_{(br,bc,cl)} cl * A[3*br+i, 3*bc+a]
                    for (br, bc, cl) in left_combo:
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * m + col_A][col] += cl
                    
                    # V: Y[a,k] = sum over right_combo entries
                    #   Y[a,k] = sum_{(br,bc,cr)} cr * B[3*br+a, 3*bc+k]
                    for (br, bc, cr) in right_combo:
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * p + col_B][col] += cr
                    
                    # W: this term contributes to output C at positions
                    # determined by which result blocks include term t_idx.
                    for (rbr, rbc) in result_contribs:
                        for (si, sign) in result_contribs[(rbr, rbc)]:
                            if si == t_idx:
                                i_total = 3 * rbr + i
                                k_total = 3 * rbc + k
                                # W row index: k*n + i (trace convention)
                                W_row = k_total * n + i_total
                                W[W_row][col] += sign
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
