"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a column
    def set_col(c, u_vals, v_vals, w_vals):
        for i, val in u_vals.items():
            U[i][c] = val
        for i, val in v_vals.items():
            V[i][c] = val
        for i, val in w_vals.items():
            W[i][c] = val

    # U indices: 0:a00, 1:a01, 2:a10, 3:a11
    # V indices: 0:b00, 1:b01, 2:b02, 3:b03, 4:b10, 5:b11, 6:b12, 7:b13
    # W indices: 0:c00, 1:c01, 2:c10, 3:c11, 4:c20, 5:c21, 6:c30, 7:c31

    # Left Block (C_L = A B_L), Cols 0-6
    # M1: (a00+a11)(b00+b11)
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    # M2: (a10+a11)b00
    set_col(1, {2:1, 3:1}, {0:1}, {2:1, 3:-1})
    # M3: a00(b10-b11)
    set_col(2, {0:1}, {4:1, 5:-1}, {1:1, 3:1})
    # M4: a11(b01-b00)
    set_col(3, {3:1}, {1:1, 0:-1}, {0:1, 2:1})
    # M5: (a00+a01)b11
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 1:1})
    # M6: (a10-a00)(b00+b01)
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    # M7: (a01-a11)(b10+b11)
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})

    # Right Block (C_R = A B_R), Cols 7-13
    # M1': (a00+a11)(b02+b13)
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    # M2': (a10+a11)b02
    set_col(8, {2:1, 3:1}, {2:1}, {6:1, 7:-1})
    # M3': a00(b12-b13)
    set_col(9, {0:1}, {6:1, 7:-1}, {5:1, 7:1})
    # M4': a11(b03-b02)
    set_col(10, {3:1}, {3:1, 2:-1}, {4:1, 6:1})
    # M5': (a00+a01)b13
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 5:1})
    # M6': (a10-a00)(b02+b03)
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    # M7': (a01-a11)(b12+b13)
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
