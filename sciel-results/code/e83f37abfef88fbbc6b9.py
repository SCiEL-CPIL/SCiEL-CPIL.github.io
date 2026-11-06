"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: One level of Strassen on 6x6 = 2x2 blocks of 3x3.
    - 7 Strassen multiplications of 3x3 matrices
    - Each 3x3 multiplication decomposed via schoolbook (rank 27)
    - Total rank = 7 * 27 = 189
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s wall-clock kill

    n = m = p = 6
    rank = 189
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    # 7 Strassen combinations at level 1
    # A = [[A00, A01],[A10, A11]], B = [[B00, B01],[B10, B11]]
    # each block is 3x3
    #
    # Strassen terms (each is a 3x3 = 3x3 multiplication):
    # T1: (A00+A11)*(B00+B11)
    # T2: (A10+A11)*B00
    # T3: A00*(B01-B11)
    # T4: A11*(B10-B00)
    # T5: (A00+A01)*B11
    # T6: (A10-A00)*(B00+B01)
    # T7: (A01-A11)*(B10+B11)
    #
    # Result blocks:
    # C00 = T1 + T4 - T5 + T7
    # C01 = T3 + T5
    # C10 = T2 + T4
    # C11 = T1 - T2 + T3 + T6

    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
    ]

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
                    # Check deadline
                    if time.time() > deadline:
                        # Return what we have so far (should be complete if we get here)
                        pass

                    for (br, bc), coeff_l in left_combo.items():
                        row_A = 3 * br + i
                        col_A = 3 * bc + a
                        U[row_A * 6 + col_A][col] += coeff_l

                    for (br, bc), coeff_r in right_combo.items():
                        row_B = 3 * br + a
                        col_B = 3 * bc + k
                        V[row_B * 6 + col_B][col] += coeff_r

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
