"""Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""

def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.

    Tensor convention:
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]

    with n = m = p = 2.

    The rank is 7, which is the proven theoretical minimum for 2x2 matrix multiplication
    (Strassen's lower bound). This is an exact decomposition with integer entries.
    """
    n, m, p = 2, 2, 2
    rank = 7

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)

    # Strassen's seven products:
    #
    # A = [[a00, a01],
    #      [a10, a11]]
    #
    # B = [[b00, b01],
    #      [b10, b11]]
    #
    # C = A * B
    #
    # m0 = (a00 + a11) * (b00 + b11)
    # m1 = (a10 + a11) * b00
    # m2 = a00 * (b01 - b11)
    # m3 = a11 * (b10 - b00)
    # m4 = (a00 + a01) * b11
    # m5 = (a10 - a00) * (b00 + b01)
    # m6 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5

    # Index mapping:
    # U row: i*m + j  => a00:0, a01:1, a10:2, a11:3
    # V row: j*p + k  => b00:0, b01:1, b10:2, b11:3
    # W row: k*n + i  => c00:0, c10:1, c01:2, c11:3

    # --- Product 0: m0 = (a00 + a11)(b00 + b11) ---
    U[0][0] = 1   # a00
    U[3][0] = 1   # a11
    V[0][0] = 1   # b00
    V[3][0] = 1   # b11
    W[0][0] = 1   # c00
    W[3][0] = 1   # c11

    # --- Product 1: m1 = (a10 + a11)b00 ---
    U[2][1] = 1   # a10
    U[3][1] = 1   # a11
    V[0][1] = 1   # b00
    W[1][1] = 1   # c10
    W[3][1] = -1  # c11

    # --- Product 2: m2 = a00(b01 - b11) ---
    U[0][2] = 1   # a00
    V[1][2] = 1   # b01
    V[3][2] = -1  # b11
    W[2][2] = 1   # c01
    W[3][2] = 1   # c11

    # --- Product 3: m3 = a11(b10 - b00) ---
    U[3][3] = 1   # a11
    V[2][3] = 1   # b10
    V[0][3] = -1  # b00
    W[0][3] = 1   # c00
    W[1][3] = 1   # c10

    # --- Product 4: m4 = (a00 + a01)b11 ---
    U[0][4] = 1   # a00
    U[1][4] = 1   # a01
    V[3][4] = 1   # b11
    W[0][4] = -1  # c00
    W[2][4] = 1   # c01

    # --- Product 5: m5 = (a10 - a00)(b00 + b01) ---
    U[2][5] = 1   # a10
    U[0][5] = -1  # a00
    V[0][5] = 1   # b00
    V[1][5] = 1   # b01
    W[3][5] = 1   # c11

    # --- Product 6: m6 = (a01 - a11)(b10 + b11) ---
    U[1][6] = 1   # a01
    U[3][6] = -1  # a11
    V[2][6] = 1   # b10
    V[3][6] = 1   # b11
    W[0][6] = 1   # c00

    return {"U": U, "V": V, "W": W}
