"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen rank-7 tensor decomposition for 2x2 matrix multiplication.
    # Index conventions:
    #   U rows: 0=A00, 1=A01, 2=A10, 3=A11
    #   V rows: 0=B00, 1=B01, 2=B10, 3=B11
    #   W rows: W[k*n+i] -> 0=C00, 1=C10, 2=C01, 3=C11
    n, m, p = DIMENSIONS
    r = 7
    U = [[0.0] * r for _ in range(4)]
    V = [[0.0] * r for _ in range(4)]
    W = [[0.0] * r for _ in range(4)]

    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0

    # M2 = (A10+A11)*B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0; W[3][1] = -1.0

    # M3 = A00*(B01-B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0; W[3][2] = 1.0

    # M4 = A11*(B10-B00)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0

    # M5 = (A00+A01)*B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0

    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0

    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
