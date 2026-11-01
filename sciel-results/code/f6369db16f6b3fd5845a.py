"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 2

    # Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2.
    # Index convention:
    #   U[i*m + j, r]  (row index = i*m + j)
    #   V[j*p + k, r]  (row index = j*p + k)
    #   W[k*n + i, r]  (row index = k*n + i)
    rank = 7

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # Strassen's 7 products:
    # M1 = (A00)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10)
    # M4 = (A11)(B01)
    # M5 = (A00+A11)(B11-B10)
    # M6 = (A01-A10)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U rows: (i,j) pairs -> indices 0,1,2,3
    # V rows: (j,k) pairs -> indices 0,1,2,3
    # W rows: (k,i) pairs -> indices 0,1,2,3

    # r=0: M1 = A00 * (B00 + B11)
    U[0][0] = 1   # A00
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11
    W[0][0] = 1   # C00
    W[3][0] = 1   # C11

    # r=1: M2 = (A10 + A11) * B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00
    W[1][1] = 1   # C10
    W[3][1] = -1  # C11

    # r=2: M3 = A00 * B10
    U[0][2] = 1   # A00
    V[2][2] = 1   # B10
    W[2][2] = 1   # C01
    W[3][2] = 1   # C11

    # r=3: M4 = A11 * B01
    U[3][3] = 1   # A11
    V[1][3] = 1   # B01
    W[0][3] = 1   # C00
    W[1][3] = 1   # C10

    # r=4: M5 = (A00 + A11) * (B11 - B10)
    U[0][4] = 1   # A00
    U[3][4] = 1   # A11
    V[3][4] = 1   # B11
    V[2][4] = -1  # -B10
    W[0][4] = -1  # C00
    W[2][4] = 1   # C01

    # r=5: M6 = (A01 - A10) * (B00 + B10)
    U[1][5] = 1   # A01
    U[2][5] = -1  # -A10
    V[0][5] = 1   # B00
    V[2][5] = 1   # B10
    W[3][5] = 1   # C11

    # r=6: M7 = (A10 - A11) * (B01 + B11)
    U[2][6] = 1   # A10
    U[3][6] = -1  # -A11
    V[1][6] = 1   # B01
    V[3][6] = 1   # B11
    W[0][6] = 1   # C00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
