import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Strassen 2x2 factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 - M6 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M5 - M3

    su = [
        [[1,0],[0,1]],   # M1
        [[0,0],[1,1]],   # M2
        [[1,0],[0,0]],   # M3
        [[0,0],[0,1]],   # M4
        [[1,1],[0,0]],   # M5
        [[-1,0],[1,0]],  # M6
        [[0,1],[0,-1]],  # M7
    ]
    sv = [
        [[1,0],[0,1]],   # M1
        [[1,0],[0,0]],   # M2
        [[0,1],[0,-1]],  # M3
        [[-1,1],[0,0]],  # M4
        [[0,0],[0,1]],   # M5
        [[1,1],[0,0]],   # M6
        [[0,1],[1,1]],   # M7
    ]
    sw = [
        [[1,0],[0,1]],   # M1
        [[0,0],[1,-1]],  # M2
        [[0,1],[0,-1]],  # M3
        [[-1,1],[0,0]],  # M4
        [[0,0],[1,1]],   # M5
        [[-1,0],[0,0]],  # M6
        [[1,0],[0,0]],   # M7
    ]

    # Verify 2x2 Strassen
    A2 = np.array([[1,2],[3,4]], dtype=float)
    B2 = np.array([[5,6],[7,8]], dtype=float)
    C2 = A2 @ B2
    C2_rec = np.zeros((2,2))
    for t in range(7):
        left = sum(A2[i][j] * su[t][i][j] for i in range(2) for j in range(2))
        right = sum(B2[i][j] * sv[t][i][j] for i in range(2) for j in range(2))
        for i in range(2):
            for j in range(2):
                C2_rec[i][j] += sw[t][i][j] * left * right
    if not np.allclose(C2_rec, C2, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Build 4x4*4x4 decomposition using recursive Strassen (49 terms)
    # Then add 4 terms for the 5th column of B.
    rank_total = 49 + 4
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    # First 49 terms: Strassen for 4x4*4x4 (columns 0-3 of B)
    for s in range(7):
        for t in range(7):
            idx = s * 7 + t
            for I in range(2):
                for J in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            ai = 2*I + i2
                            aj = 2*J + j2
                            U[ai * m + aj, idx] = su[s][I][J] * su[t][i2][j2]
                            V[ai * p + aj, idx] = sv[s][I][J] * sv[t][i2][j2]
                            W[aj * n + ai, idx] = sw[s][I][J] * sw[t][i2][j2]

    # Last 4 terms: for the 5th column of B (column index 4)
    for i in range(4):
        idx = 49 + i
        for j in range(4):
            U[i * m + j, idx] = 1.0
            V[j * p + 4, idx] = 1.0
            W[4 * n + i, idx] = 1.0

    # Verify with random test
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 5)
    C_test = A_test @ B_test

    C_recon = np.zeros((4, 5))
    for t in range(rank_total):
        U_t = U[:, t].reshape(4, 4)
        V_t = V[:, t].reshape(4, 5)
        W_t = W[:, t].reshape(5, 4)  # W[k*n+i] -> W_t[k, i]
        left = np.sum(A_test * U_t)
        right = np.sum(B_test * V_t)
        C_recon += left * right * W_t.T

    if not np.allclose(C_recon, C_test, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
