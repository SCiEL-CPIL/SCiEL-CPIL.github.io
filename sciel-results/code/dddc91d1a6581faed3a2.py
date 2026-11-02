"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen for first 2 columns of B (indices 0, 1 in V's second dim, 0, 1 in W's first dim)
    # V indices: 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23
    # W indices: 0:c11, 1:c21, 2:c12, 3:c22, 4:c13, 5:c23
    # U indices: 0:a11, 1:a12, 2:a21, 3:a22

    # Col 0: m1 = (a11+a22)(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # Col 1: m2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1

    # Col 2: m3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # Col 3: m4 = a22(b21-b11)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # Col 4: m5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # Col 5: m6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # Col 6: m7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1

    # Schoolbook for 3rd column (indices 2 in V, 4,5 in W)
    # c13 = a11*b13 + a12*b23
    # c23 = a21*b13 + a22*b23

    # Col 7: a11*b13
    U[0][7] = 1
    V[2][7] = 1
    W[4][7] = 1

    # Col 8: a12*b23
    U[1][8] = 1
    V[5][8] = 1
    W[4][8] = 1

    # Col 9: a21*b13
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1

    # Col 10: a22*b23
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
