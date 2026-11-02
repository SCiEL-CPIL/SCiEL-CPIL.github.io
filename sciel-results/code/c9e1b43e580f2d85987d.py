"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen bilinear forms definitions
    # Each form is a tuple: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: [a00, a01, a10, a11]
    # b_coeffs: [b00, b01, b10, b11]
    # c_coeffs: [c00, c01, c10, c11]
    
    strassen_forms = [
        # P1 = (a00 + a11)(b00 + b11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2 = (a10 + a11)b00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # c01 += 1, c11 -= 1
        # P3 = a00(b10 - b11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),  # c01 += 1, c11 += 1
        # P4 = a11(b01 - b00)
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 0, 0]),  # c00 += 1
        # P5 = (a00 + a10)b11
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # c00 -= 1, c10 += 1
        # P6 = (a11 - a00)(b00 + b01)
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]),  # c10 += 1, c11 += 1
        # P7 = (a01 - a11)(b10 + b11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # c00 += 1
    ]
    
    # Define the 4 blocks
    # Block 1: A cols 0,1; B rows 0,1; B cols 0,1
    # Block 2: A cols 0,1; B rows 0,1; B cols 2,3
    # Block 3: A cols 2,3; B rows 2,3; B cols 0,1
    # Block 4: A cols 2,3; B rows 2,3; B cols 2,3
    blocks = [
        (0, 1, 0, 1, 0, 1), # a_col_start, a_col_end, b_row_start, b_row_end, b_col_start, b_col_end
        (0, 1, 0, 1, 2, 3),
        (2, 3, 2, 3, 0, 1),
        (2, 3, 2, 3, 2, 3),
    ]
    
    col = 0
    for a_cs, a_ce, b_rs, b_re, b_cs, b_ce in blocks:
        # Local A indices mapping to global
        # a00 -> A[0, a_cs], a01 -> A[0, a_ce], a10 -> A[1, a_cs], a11 -> A[1, a_ce]
        a_indices = [
            0 * m + a_cs, # a00
            0 * m + a_ce, # a01
            1 * m + a_cs, # a10
            1 * m + a_ce  # a11
        ]
        
        # Local B indices mapping to global
        # b00 -> B[b_rs, b_cs], b01 -> B[b_rs, b_ce], b10 -> B[b_re, b_cs], b11 -> B[b_re, b_ce]
        b_indices = [
            b_rs * p + b_cs, # b00
            b_rs * p + b_ce, # b01
            b_re * p + b_cs, # b10
            b_re * p + b_ce  # b11
        ]
        
        # Local C indices mapping to global
        # c00 -> C[0, b_cs], c01 -> C[0, b_ce], c10 -> C[1, b_cs], c11 -> C[1, b_ce]
        # W index is k*n + i
        c_indices = [
            b_cs * n + 0, # c00
            b_ce * n + 0, # c01
            b_cs * n + 1, # c10
            b_ce * n + 1  # c11
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_forms:
            for i in range(4):
                if a_coeffs[i] != 0:
                    U[a_indices[i]][col] += a_coeffs[i]
                if b_coeffs[i] != 0:
                    V[b_indices[i]][col] += b_coeffs[i]
                if c_coeffs[i] != 0:
                    W[c_indices[i]][col] += c_coeffs[i]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
