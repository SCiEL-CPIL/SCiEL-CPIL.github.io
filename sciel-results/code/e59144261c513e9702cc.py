"""Rank-minimal tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7)."""
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.

    Tensor convention (official trace convention):
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]

    with n = m = p = 2.

    Strassen's 7 products:
        m0 = (a00 + a11) * (b00 + b11)
        m1 = (a10 + a11) * b00
        m2 = a00 * (b01 - b11)
        m3 = a11 * (b10 - b00)
        m4 = (a00 + a01) * b11
        m5 = (a10 - a00) * (b00 + b01)
        m6 = (a01 - a11) * (b10 + b11)

    Reconstruction:
        c00 = m0 + m3 - m4 + m6
        c01 = m2 + m4
        c10 = m1 + m3
        c11 = m0 - m1 + m2 + m5

    Rank 7 is the proven minimum for 2x2 matrix multiplication over R.
    """
    start_time = time.time()
    deadline = start_time + 119  # Safety margin

    n, m, p = 2, 2, 2
    rank = 7

    # U: shape (n*m, rank) = (4, 7), indexed by [i*m + j, c]
    # V: shape (m*p, rank) = (4, 7), indexed by [j*p + k, c]
    # W: shape (p*n, rank) = (4, 7), indexed by [k*n + i, c]

    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]

    # Product 0: m0 = (a00 + a11)(b00 + b11)
    U[0][0] = 1;  U[3][0] = 1
    V[0][0] = 1;  V[3][0] = 1
    W[0][0] = 1;  W[3][0] = 1

    # Product 1: m1 = (a10 + a11)b00
    U[2][1] = 1;  U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1;  W[3][1] = -1

    # Product 2: m2 = a00(b01 - b11)
    U[0][2] = 1
    V[1][2] = 1;  V[3][2] = -1
    W[2][2] = 1;  W[3][2] = 1

    # Product 3: m3 = a11(b10 - b00)
    U[3][3] = 1
    V[2][3] = 1;  V[0][3] = -1
    W[0][3] = 1;  W[1][3] = 1

    # Product 4: m4 = (a00 + a01)b11
    U[0][4] = 1;  U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1

    # Product 5: m5 = (a10 - a00)(b00 + b01)
    U[2][5] = 1;  U[0][5] = -1
    V[0][5] = 1;  V[1][5] = 1
    W[3][5] = 1

    # Product 6: m6 = (a01 - a11)(b10 + b11)
    U[1][6] = 1;  U[3][6] = -1
    V[2][6] = 1;  V[3][6] = 1
    W[0][6] = 1

    # Deadline guard: return immediately if we're close to the limit
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
