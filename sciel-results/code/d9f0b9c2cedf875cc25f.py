import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Rank 47: 7 Strassen terms for 2x2x2 block + 40 schoolbook for remaining
    R = 47

    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Strassen 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*m + j, V index: j*p + k, W index: k*n + i
    # C00: W[0], C01: W[1], C10: W[3], C11: W[4]

    # Term 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1   # A00: i=0,j=0
    U[5, 0] = 1   # A11: i=1,j=1
    V[0, 0] = 1   # B00: j=0,k=0
    V[5, 0] = 1   # B11: j=1,k=1
    W[0, 0] = 1   # C00
    W[4, 0] = 1   # C11

    # Term 1: M2 = (A10+A11)B00
    U[4, 1] = 1   # A10: i=1,j=0
    U[5, 1] = 1   # A11: i=1,j=1
    V[0, 1] = 1   # B00: j=0,k=0
    W[3, 1] = 1   # C10

    # Term 2: M3 = A00(B10-B11)
    U[0, 2] = 1   # A00: i=0,j=0
    V[4, 2] = 1   # B10: j=1,k=0
    V[5, 2] = -1  # B11: j=1,k=1
    W[1, 2] = 1   # C01

    # Term 3: M4 = A11(B01-B00)
    U[5, 3] = 1   # A11: i=1,j=1
    V[1, 3] = 1   # B01: j=0,k=1
    V[0, 3] = -1  # B00: j=0,k=0
    W[0, 3] = -1  # C00
    W[1, 3] = 1   # C01

    # Term 4: M5 = (A00+A10)B11
    U[0, 4] = 1   # A00: i=0,j=0
    U[4, 4] = 1   # A10: i=1,j=0
    V[5, 4] = 1   # B11: j=1,k=1
    W[0, 4] = 1   # C00
    W[3, 4] = 1   # C10

    # Term 5: M6 = (A10-A00)(B00+B10)
    U[4, 5] = 1   # A10: i=1,j=0
    U[0, 5] = -1  # A00: i=0,j=0
    V[0, 5] = 1   # B00: j=0,k=0
    V[4, 5] = 1   # B10: j=1,k=0
    W[3, 5] = 1   # C10
    W[4, 5] = 1   # C11

    # Term 6: M7 = (A11-A01)(B01+B11)
    U[5, 6] = 1   # A11: i=1,j=1
    U[1, 6] = -1  # A01: i=0,j=1
    V[1, 6] = 1   # B01: j=0,k=1
    V[5, 6] = 1   # B11: j=1,k=1
    W[0, 6] = -1  # C00
    W[1, 6] = 1   # C01
    W[4, 6] = 1   # C11

    # Fill remaining terms with schoolbook (skip 2x2x2 block)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        Ua = np.array(U_mat, dtype=float)
        Va = np.array(V_mat, dtype=float)
        Wa = np.array(W_mat, dtype=float)
        rank = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k2 * n + i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n * m, rank_sb))
        V_sb = np.zeros((m * p, rank_sb))
        W_sb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1
                    V_sb[j * p + k, col] = 1
                    W_sb[k * n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
