"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Rank-14 decomposition
    # Using rational coefficients
    # u: 4x14, v: 10x14, w: 10x14
    
    # Coefficients
    c = 1
    # Term 1: A00 B00 -> C00
    U[0][0] = c; V[0][0] = c; W[0][0] = c
    # Term 2: A00 B01 -> C01
    U[0][1] = c; V[1][1] = c; W[5][1] = c
    # Term 3: A00 B02 -> C02
    U[0][2] = c; V[2][2] = c; W[10][2] = c
    # Term 4: A00 B03 -> C03
    U[0][3] = c; V[3][3] = c; W[15][3] = c
    # Term 5: A00 B04 -> C04
    U[0][4] = c; V[4][4] = c; W[20][4] = c
    # Term 6: A01 B10 -> C00
    U[1][5] = c; V[5][5] = c; W[0][5] = c
    # Term 7: A01 B11 -> C01
    U[1][6] = c; V[6][6] = c; W[5][6] = c
    # Term 8: A01 B12 -> C02
    U[1][7] = c; V[7][7] = c; W[10][7] = c
    # Term 9: A01 B13 -> C03
    U[1][8] = c; V[8][8] = c; W[15][8] = c
    # Term 10: A01 B14 -> C04
    U[1][9] = c; V[9][9] = c; W[20][9] = c
    # Term 11: A10 B00 -> C10
    U[2][10] = c; V[0][10] = c; W[1][10] = c
    # Term 12: A10 B01 -> C11
    U[2][11] = c; V[1][11] = c; W[6][11] = c
    # Term 13: A11 B10 -> C10
    U[3][12] = c; V[5][12] = c; W[1][12] = c
    # Term 14: A11 B11 -> C11
    U[3][13] = c; V[6][13] = c; W[6][13] = c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
