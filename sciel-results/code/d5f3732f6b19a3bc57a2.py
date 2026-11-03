"""Optimized program: Kronecker product of Strassen (2x2x2, rank 7) and 1x2x3 (rank 6) decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6

    # Strassen decomposition of 2x2x2 (rank 7)
    # Ux: 4x7, Vx: 4x7, Wx: 4x7
    # Ux[i*2+j, r], Vx[j*2+k, r], Wx[k*2+i, r]
    Ux = [[0]*7 for _ in range(4)]
    Vx = [[0]*7 for _ in range(4)]
    Wx = [[0]*7 for _ in range(4)]

    # Product 0: (A00+A11)(B00+B11)
    Ux[0][0] = 1; Ux[3][0] = 1
    Vx[0][0] = 1; Vx[3][0] = 1
    Wx[0][0] = 1; Wx[3][0] = 1

    # Product 1: (A10+A11)B00
    Ux[2][1] = 1; Ux[3][1] = 1
    Vx[0][1] = 1
    Wx[1][1] = 1; Wx[3][1] = 1

    # Product 2: A00(B01-B11)
    Ux[0][2] = 1
    Vx[1][2] = 1; Vx[3][2] = -1
    Wx[2][2] = 1

    # Product 3: A11(B10-B00)
    Ux[3][3] = 1
    Vx[2][3] = 1; Vx[0][3] = -1
    Wx[0][3] = 1; Wx[1][3] = 1

    # Product 4: (A00+A01)B11
    Ux[0][4] = 1; Ux[1][4] = 1
    Vx[3][4] = 1
    Wx[0][4] = -1; Wx[2][4] = 1

    # Product 5: (A10-A00)(B00+B01)
    Ux[2][5] = 1; Ux[0][5] = -1
    Vx[0][5] = 1; Vx[1][5] = 1
    Wx[3][5] = 1

    # Product 6: (A01-A11)(B10+B11)
    Ux[1][6] = 1; Ux[3][6] = -1
    Vx[2][6] = 1; Vx[3][6] = 1
    Wx[0][6] = 1

    # 1x2x3 decomposition (rank 6)
    # Uy: 2x6, Vy: 6x6, Wy: 3x6
    Uy = [[0]*6 for _ in range(2)]
    Vy = [[0]*6 for _ in range(6)]
    Wy = [[0]*6 for _ in range(3)]

    for j in range(2):
        for k in range(3):
            r = j*3 + k
            Uy[j][r] = 1
            Vy[j*3+k][r] = 1
            Wy[k][r] = 1

    # Kronecker product: rank = 7*6 = 42
    rank = 42
    U = [[0]*rank for _ in range(8)]   # n*m = 8 rows
    V = [[0]*rank for _ in range(24)]  # m*p = 24 rows
    W = [[0]*rank for _ in range(12)]  # p*n = 12 rows

    # Index mapping:
    # i = i1 (i1 in {0,1})
    # j = j1*2 + j2 (j1 in {0,1}, j2 in {0,1})
    # k = k1*3 + k2 (k1 in {0,1}, k2 in {0,1,2})

    for r1 in range(7):
        for r2 in range(6):
            r = r1*6 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(3):
                                i = i1
                                j = j1*2 + j2
                                k = k1*3 + k2

                                u_val = Ux[i1*2 + j1][r1] * Uy[j2][r2]
                                if u_val != 0:
                                    U[i*4 + j][r] += u_val

                                v_val = Vx[j1*2 + k1][r1] * Vy[j2*3 + k2][r2]
                                if v_val != 0:
                                    V[j*6 + k][r] += v_val

                                w_val = Wx[k1*2 + i1][r1] * Wy[k2][r2]
                                if w_val != 0:
                                    W[k*2 + i][r] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
