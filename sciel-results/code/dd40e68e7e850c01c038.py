"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = 2, 4, 6
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # 6 sub-multiplications: (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
    subs = [
        (0, 0, 0, 0, 0, 0),  # A_L * B_11 -> C[0:2, 0:2]
        (0, 2, 2, 0, 0, 0),  # A_R * B_21 -> C[0:2, 0:2]
        (0, 0, 0, 2, 0, 2),  # A_L * B_12 -> C[0:2, 2:4]
        (0, 2, 2, 2, 0, 2),  # A_R * B_22 -> C[0:2, 2:4]
        (0, 0, 0, 4, 0, 4),  # A_L * B_13 -> C[0:2, 4:6]
        (0, 2, 2, 4, 0, 4),  # A_R * B_23 -> C[0:2, 4:6]
    ]

    # Strassen 7 terms: (u_coeffs [x00,x01,x10,x11], v_coeffs [y00,y01,y10,y11], w_coeffs [z00,z01,z10,z11])
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),   # m2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # m3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # m4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # m6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # m7
    ]

    col = 0
    for (ar0, ac0, br0, bc0, cr0, cc0) in subs:
        # Index mappings
        x_idx = [[ar0 * m + ac0, ar0 * m + ac0 + 1],
                 [(ar0 + 1) * m + ac0, (ar0 + 1) * m + ac0 + 1]]
        y_idx = [[br0 * p + bc0, br0 * p + bc0 + 1],
                 [(br0 + 1) * p + bc0, (br0 + 1) * p + bc0 + 1]]
        # W[k*n+i] where k=col, i=row
        z_idx = [[cc0 * n + cr0, (cc0 + 1) * n + cr0],
                 [cc0 * n + cr0 + 1, (cc0 + 1) * n + cr0 + 1]]

        for (u_c, v_c, w_c) in strassen_terms:
            for li in range(2):
                for lj in range(2):
                    if u_c[li * 2 + lj] != 0:
                        U[x_idx[li][lj]][col] += u_c[li * 2 + lj]
                    if v_c[li * 2 + lj] != 0:
                        V[y_idx[li][lj]][col] += v_c[li * 2 + lj]
                    if w_c[li * 2 + lj] != 0:
                        W[z_idx[li][lj]][col] += w_c[li * 2 + lj]
            col += 1

        if time.time() > DEADLINE:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
