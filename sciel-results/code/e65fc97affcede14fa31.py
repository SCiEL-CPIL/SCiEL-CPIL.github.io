import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2x2.
    U: 4x7, V: 4x7, W: 4x7
    A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1: A00 * B00
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11

    # M2: (A10 + A11) * B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1   # C10
    W[3, 1] = -1  # C11

    # M3: A00 * B01
    U[0, 2] = 1
    V[1, 2] = 1
    W[1, 2] = 1  # C01

    # M4: A11 * B11
    U[3, 3] = 1
    V[3, 3] = 1
    W[0, 3] = 1  # C00
    W[3, 3] = 1  # C11

    # M5: (A00 + A11) * (B10 + B11)
    U[0, 4] = 1
    U[3, 4] = 1
    V[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00
    W[2, 4] = 1   # C10

    # M6: (A01 - A11) * (B10 + B11)
    U[1, 5] = 1
    U[3, 5] = -1
    V[2, 5] = 1
    V[3, 5] = 1
    W[1, 5] = 1  # C01
    W[3, 5] = 1  # C11

    # M7: (A10 - A00) * (B00 + B01)
    U[2, 6] = 1
    U[0, 6] = -1
    V[0, 6] = 1
    V[1, 6] = 1
    W[0, 6] = 1  # C00

    return U, V, W

def _block_2x3x2():
    """
    Decompose 2x3x2 using Strassen for 2x2x2 + trivial for 2x1x2.
    Rank = 7 + 4 = 11.
    Returns U (6x11), V (6x11), W (4x11)
    """
    U_s, V_s, W_s = _strassen_2x2x2()

    U_t = np.zeros((2, 4))
    V_t = np.zeros((2, 4))
    W_t = np.zeros((4, 4))
    for i in range(2):
        for k in range(2):
            U_t[i, i * 2 + k] = 1
            V_t[k, i * 2 + k] = 1
            W_t[k * 2 + i, i * 2 + k] = 1

    rank = 11
    U_full = np.zeros((6, rank))
    V_full = np.zeros((6, rank))
    W_full = np.zeros((4, rank))

    # Strassen part (columns 0-6)
    for i in range(2):
        for j in range(2):
            U_full[i * 3 + j, 0:7] = U_s[i * 2 + j, 0:7]
    for j in range(2):
        for k in range(2):
            V_full[j * 2 + k, 0:7] = V_s[j * 2 + k, 0:7]
    for k in range(2):
        for i in range(2):
            W_full[k * 2 + i, 0:7] = W_s[k * 2 + i, 0:7]

    # Trivial part (columns 7-10)
    for i in range(2):
        U_full[i * 3 + 2, 7:11] = U_t[i, 0:4]
    for k in range(2):
        V_full[2 * 2 + k, 7:11] = V_t[k, 0:4]
    for k in range(2):
        for i in range(2):
            W_full[k * 2 + i, 7:11] = W_t[k * 2 + i, 0:4]

    return U_full, V_full, W_full

def _block_2x3x1():
    """
    Trivial decomposition for 2x3x1.
    Rank = 6.
    Returns U (6x6), V (3x6), W (2x6)
    """
    rank = 6
    U = np.zeros((6, rank))
    V = np.zeros((3, rank))
    W = np.zeros((2, rank))
    col = 0
    for i in range(2):
        for j in range(3):
            U[i * 3 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _hybrid_decomposition_2x3x7():
    """
    Hybrid Strassen/Schoolbook decomposition for 2x3x7.
    Split 7 columns into 2+2+2+1.
    Rank = 11 + 11 + 11 + 6 = 39.
    """
    n, m, p = 2, 3, 7
    rank = 39

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Block 1: columns 0-1 of B (2x3x2)
    U1, V1, W1 = _block_2x3x2()
    U[:, 0:11] = U1
    for j in range(2):
        for k in range(2):
            V[j * 7 + k, 0:11] = V1[j * 2 + k, 0:11]
    for k in range(2):
        for i in range(2):
            W[k * 2 + i, 0:11] = W1[k * 2 + i, 0:11]

    # Block 2: columns 2-3 of B (2x3x2)
    U2, V2, W2 = _block_2x3x2()
    U[:, 11:22] = U2
    for j in range(2):
        for k in range(2):
            V[j * 7 + (2 + k), 11:22] = V2[j * 2 + k, 0:11]
    for k in range(2):
        for i in range(2):
            W[k * 2 + i, 11:22] = W2[k * 2 + i, 0:11]

    # Block 3: columns 4-5 of B (2x3x2)
    U3, V3, W3 = _block_2x3x2()
    U[:, 22:33] = U3
    for j in range(2):
        for k in range(2):
            V[j * 7 + (4 + k), 22:33] = V3[j * 2 + k, 0:11]
    for k in range(2):
        for i in range(2):
            W[k * 2 + i, 22:33] = W3[k * 2 + i, 0:11]

    # Block 4: column 6 of B (2x3x1)
    U4, V4, W4 = _block_2x3x1()
    U[:, 33:39] = U4
    for j in range(3):
        V[j * 7 + 6, 33:39] = V4[j, 0:6]
    for k in range(1):
        for i in range(2):
            W[k * 2 + i, 33:39] = W4[i, 0:6]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_exact(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact for the matrix multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kin->ijk', U_3d, V_3d, W_3d)

    if np.allclose(T, 1.0, atol=1e-9):
        return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin well under 150s

    n, m, p = DIMENSIONS

    # Try hybrid decomposition first
    try:
        hybrid = _hybrid_decomposition_2x3x7()
        if time.time() < deadline and _verify_exact(hybrid["U"], hybrid["V"], hybrid["W"], n, m, p):
            return hybrid
    except Exception:
        pass

    # Fall back to trivial
    return _trivial_decomposition(n, m, p)
