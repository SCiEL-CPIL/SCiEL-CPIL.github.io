"""Strassen rank-7 exact decomposition for 2x2 matrix multiplication."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm gives an exact rank-7 decomposition for 2x2 matmul.
    # Input A is 2x2 with entries A[i*m + j] = A[i,j]  (m=2)
    # Input B is 2x2 with entries B[j*p + k] = B[j,k]  (p=2)
    # Output C follows trace convention W[k*n + i, r] => C[i,k] (n=2)
    #
    # Strassen factors:
    #   M1 = (A00 + A11) * (B00 + B11)
    #   M2 = (A10 + A11) * B00
    #   M3 = A00 * (B01 - B11)
    #   M4 = A11 * (B10 - B00)
    #   M5 = (A00 + A01) * B11
    #   M6 = (A10 - A00) * (B00 + B01)
    #   M7 = (A01 - A11) * (B10 + B11)
    #
    # Output:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # U has shape (n*m, rank) = (4, 7): rows indexed by i*m+j
    # V has shape (m*p, rank) = (4, 7): rows indexed by j*p+k
    # W has shape (p*n, rank) = (4, 7): rows indexed by k*n+i
    #
    # Index mapping:
    #   A00 -> row 0, A01 -> row 1, A10 -> row 2, A11 -> row 3
    #   B00 -> row 0, B01 -> row 1, B10 -> row 2, B11 -> row 3
    #   C00 -> row 0 (k=0,i=0), C01 -> row 1 (k=0,i=1)
    #   C10 -> row 2 (k=1,i=0), C11 -> row 3 (k=1,i=1)

    rank = 7

    # U[i*m + j, c] coefficients for each factor c
    # Factor 1: (A00 + A11) * (B00 + B11)
    # Factor 2: (A10 + A11) * B00
    # Factor 3: A00 * (B01 - B11)
    # Factor 4: A11 * (B10 - B00)
    # Factor 5: (A00 + A01) * B11
    # Factor 6: (A10 - A00) * (B00 + B01)
    # Factor 7: (A01 - A11) * (B10 + B11)

    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # Factor 1: (A00 + A11) * (B00 + B11)
    U[0][0] = 1   # A00
    U[3][0] = 1   # A11
    V[0][0] = 1   # B00
    V[3][0] = 1   # B11

    # Factor 2: (A10 + A11) * B00
    U[2][1] = 1   # A10
    U[3][1] = 1   # A11
    V[0][1] = 1   # B00

    # Factor 3: A00 * (B01 - B11)
    U[0][2] = 1   # A00
    V[1][2] = 1   # B01
    V[3][2] = -1  # B11

    # Factor 4: A11 * (B10 - B00)
    U[3][3] = 1   # A11
    V[2][3] = 1   # B10
    V[0][3] = -1  # B00

    # Factor 5: (A00 + A01) * B11
    U[0][4] = 1   # A00
    U[1][4] = 1   # A01
    V[3][4] = 1   # B11

    # Factor 6: (A10 - A00) * (B00 + B01)
    U[2][5] = 1   # A10
    U[0][5] = -1  # A00
    V[0][5] = 1   # B00
    V[1][5] = 1   # B01

    # Factor 7: (A01 - A11) * (B10 + B11)
    U[1][6] = 1   # A01
    U[3][6] = -1  # A11
    V[2][6] = 1   # B10
    V[3][6] = 1   # B11

    # W[k*n + i, c]: C00=row0, C01=row1, C10=row2, C11=row3
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1   # M1
    W[0][3] = 1   # M4
    W[0][4] = -1  # M5
    W[0][6] = 1   # M7

    # C01 = M3 + M5
    W[1][2] = 1   # M3
    W[1][4] = 1   # M5

    # C10 = M2 + M4
    W[2][1] = 1   # M2
    W[2][3] = 1   # M4

    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1   # M1
    W[3][1] = -1  # M2
    W[3][2] = 1   # M3
    W[3][5] = 1   # M6

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
