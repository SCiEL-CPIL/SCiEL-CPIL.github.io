"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0

    def strassen_2x2x2(A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        nonlocal col
        a = A_rows[0] * m + A_cols[0]
        b = A_rows[0] * m + A_cols[1]
        c_ = A_rows[1] * m + A_cols[0]
        d = A_rows[1] * m + A_cols[1]

        e = B_rows[0] * p + B_cols[0]
        f = B_rows[0] * p + B_cols[1]
        g = B_rows[1] * p + B_cols[0]
        h = B_rows[1] * p + B_cols[1]

        # M1 = (a+d)(e+h)
        U[a][col] = 1.0; U[d][col] = 1.0
        V[e][col] = 1.0; V[h][col] = 1.0
        W[C_cols[0]*n + C_rows[0]][col] = 1.0
        W[C_cols[1]*n + C_rows[1]][col] = 1.0
        col += 1

        # M2 = (c+d)e
        U[c_][col] = 1.0; U[d][col] = 1.0
        V[e][col] = 1.0
        W[C_cols[0]*n + C_rows[1]][col] = 1.0
        W[C_cols[1]*n + C_rows[1]][col] = -1.0
        col += 1

        # M3 = a(f-h)
        U[a][col] = 1.0
        V[f][col] = 1.0; V[h][col] = -1.0
        W[C_cols[1]*n + C_rows[0]][col] = 1.0
        W[C_cols[1]*n + C_rows[1]][col] = 1.0
        col += 1

        # M4 = d(g-e)
        U[d][col] = 1.0
        V[g][col] = 1.0; V[e][col] = -1.0
        W[C_cols[0]*n + C_rows[0]][col] = 1.0
        W[C_cols[0]*n + C_rows[1]][col] = 1.0
        col += 1

        # M5 = (a+b)h
        U[a][col] = 1.0; U[b][col] = 1.0
        V[h][col] = 1.0
        W[C_cols[0]*n + C_rows[0]][col] = -1.0
        W[C_cols[1]*n + C_rows[0]][col] = 1.0
        col += 1

        # M6 = (c-d)(f+g)
        U[c_][col] = 1.0; U[d][col] = -1.0
        V[f][col] = 1.0; V[g][col] = 1.0
        W[C_cols[1]*n + C_rows[1]][col] = 1.0
        col += 1

        # M7 = (b-d)(e+g)
        U[b][col] = 1.0; U[d][col] = -1.0
        V[e][col] = 1.0; V[g][col] = 1.0
        W[C_cols[0]*n + C_rows[0]][col] = 1.0
        col += 1

    # Block 1: C cols 0,1
    strassen_2x2x2([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    strassen_2x2x2([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])

    # Block 2: C cols 2,3
    strassen_2x2x2([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    strassen_2x2x2([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])

    # Block 3: C col 4, schoolbook
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1.0
            V[j*p + 4][col] = 1.0
            W[4*n + i][col] = 1.0
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
