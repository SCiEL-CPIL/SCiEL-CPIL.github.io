"""
Tensor decomposition for 2x2 matrix multiplication (Strassen, rank 7).
The rank of the 2x2 matrix multiplication tensor is 7, which is the theoretical minimum.
The naive algorithm has rank 8.
Score = 8 / 7 = 1.1428...
"""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns Strassen's rank-7 decomposition for 2x2 matrix multiplication.

    Tensor convention:
        U[i*m + j, c]  for A[i, j]
        V[j*p + k, c]  for B[j, k]
        W[k*n + i, c]  for C[i, k]

    with n = m = p = 2.
    
    Indices:
    A: i in {0,1}, j in {0,1} -> U row i*2+j
    B: j in {0,1}, k in {0,1} -> V row j*2+k
    C: k in {0,1}, i in {0,1} -> W row k*2+i (Trace convention: W[k*n+i])
    
    C[0,0] -> W[0]
    C[0,1] -> W[2]
    C[1,0] -> W[1]
    C[1,1] -> W[3]
    """
    n, m, p = DIMENSIONS
    rank = 7

    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's seven products:
    #
    # Let A = [[a00, a01],
    #          [a10, a11]]
    #
    #     B = [[b00, b01],
    #          [b10, b11]]
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

    # --- Product 0: m0 = (a00 + a11)(b00 + b11) ---
    # U: a00 (idx 0), a11 (idx 3)
    U[0][0] = 1
    U[3][0] = 1
    # V: b00 (idx 0), b11 (idx 3)
    V[0][0] = 1
    V[3][0] = 1
    # W: c00 (idx 0), c11 (idx 3)
    W[0][0] = 1
    W[3][0] = 1

    # --- Product 1: m1 = (a10 + a11)b00 ---
    # U: a10 (idx 2), a11 (idx 3)
    U[2][1] = 1
    U[3][1] = 1
    # V: b00 (idx 0)
    V[0][1] = 1
    # W: c10 (idx 1), c11 (idx 3)
    W[1][1] = 1
    W[3][1] = -1

    # --- Product 2: m2 = a00(b01 - b11) ---
    # U: a00 (idx 0)
    U[0][2] = 1
    # V: b01 (idx 1), b11 (idx 3)
    V[1][2] = 1
    V[3][2] = -1
    # W: c01 (idx 2), c11 (idx 3)
    W[2][2] = 1
    W[3][2] = 1

    # --- Product 3: m3 = a11(b10 - b00) ---
    # U: a11 (idx 3)
    U[3][3] = 1
    # V: b10 (idx 2), b00 (idx 0)
    V[2][3] = 1
    V[0][3] = -1
    # W: c00 (idx 0), c10 (idx 1)
    W[0][3] = 1
    W[1][3] = 1

    # --- Product 4: m4 = (a00 + a01)b11 ---
    # U: a00 (idx 0), a01 (idx 1)
    U[0][4] = 1
    U[1][4] = 1
    # V: b11 (idx 3)
    V[3][4] = 1
    # W: c00 (idx 0), c01 (idx 2)
    W[0][4] = -1
    W[2][4] = 1

    # --- Product 5: m5 = (a10 - a00)(b00 + b01) ---
    # U: a10 (idx 2), a00 (idx 0)
    U[2][5] = 1
    U[0][5] = -1
    # V: b00 (idx 0), b01 (idx 1)
    V[0][5] = 1
    V[1][5] = 1
    # W: c11 (idx 3)
    W[3][5] = 1

    # --- Product 6: m6 = (a01 - a11)(b10 + b11) ---
    # U: a01 (idx 1), a11 (idx 3)
    U[1][6] = 1
    U[3][6] = -1
    # V: b10 (idx 2), b11 (idx 3)
    V[2][6] = 1
    V[3][6] = 1
    # W: c00 (idx 0)
    W[0][6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
