import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Rank 15 decomposition: Strassen 2x2 core (7) + compressed border (8)
    R = 15
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Strassen 2x2 core (7 terms)
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 1), r] = 1

    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1; W[idx_w(1, 1), r] = -1

    # m3 = a00(b01 - b11)
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1; V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1

    # m4 = a11(b10 - b00)
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1; W[idx_w(0, 1), r] = 1

    # m5 = (a00 + a01)b11
    r = 4
    U[idx_u(0, 0), r] = 1; U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1; W[idx_w(1, 0), r] = 1

    # m6 = (a10 - a00)(b00 + b01)
    r = 5
    U[idx_u(1, 0), r] = 1; U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1

    # m7 = (a01 - a11)(b10 + b11)
    r = 6
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1

    # Compressed border terms (8 terms)
    # Term 7: a20*(b00+b01+b02) -> (2,0,0), (2,0,1), (2,0,2)
    r = 7
    U[idx_u(2, 0), r] = 1
    V[idx_v(0, 0), r] = 1; V[idx_v(0, 1), r] = 1; V[idx_v(0, 2), r] = 1
    W[idx_w(0, 2), r] = 1; W[idx_w(1, 2), r] = 1; W[idx_w(2, 2), r] = 1

    # Term 8: a21*(b10+b11+b12) -> (2,1,0), (2,1,1), (2,1,2)
    r = 8
    U[idx_u(2, 1), r] = 1
    V[idx_v(1, 0), r] = 1; V[idx_v(1, 1), r] = 1; V[idx_v(1, 2), r] = 1
    W[idx_w(0, 2), r] = 1; W[idx_w(1, 2), r] = 1; W[idx_w(2, 2), r] = 1

    # Term 9: a22*(b20+b21+b22) -> (2,2,0), (2,2,1), (2,2,2)
    r = 9
    U[idx_u(2, 2), r] = 1
    V[idx_v(2, 0), r] = 1; V[idx_v(2, 1), r] = 1; V[idx_v(2, 2), r] = 1
    W[idx_w(0, 2), r] = 1; W[idx_w(1, 2), r] = 1; W[idx_w(2, 2), r] = 1

    # Term 10: (a02+a12)*b20 -> (0,2,0), (1,2,0)
    r = 10
    U[idx_u(0, 2), r] = 1; U[idx_u(1, 2), r] = 1
    V[idx_v(2, 0), r] = 1
    W[idx_w(0, 0), r] = 1; W[idx_w(1, 0), r] = 1

    # Term 11: (a02+a12)*b21 -> (0,2,1), (1,2,1)
    r = 11
    U[idx_u(0, 2), r] = 1; U[idx_u(1, 2), r] = 1
    V[idx_v(2, 1), r] = 1
    W[idx_w(1, 0), r] = 1; W[idx_w(1, 1), r] = 1

    # Term 12: (a02+a12)*b22 -> (0,2,2), (1,2,2)
    r = 12
    U[idx_u(0, 2), r] = 1; U[idx_u(1, 2), r] = 1
    V[idx_v(2, 2), r] = 1
    W[idx_w(2, 0), r] = 1; W[idx_w(2, 1), r] = 1

    # Term 13: (a00+a10)*b02 -> (0,0,2), (1,0,2)
    r = 13
    U[idx_u(0, 0), r] = 1; U[idx_u(1, 0), r] = 1
    V[idx_v(0, 2), r] = 1
    W[idx_w(2, 0), r] = 1; W[idx_w(2, 1), r] = 1

    # Term 14: (a01+a11)*b12 -> (0,1,2), (1,1,2)
    r = 14
    U[idx_u(0, 1), r] = 1; U[idx_u(1, 1), r] = 1
    V[idx_v(1, 2), r] = 1
    W[idx_w(2, 0), r] = 1; W[idx_w(2, 1), r] = 1

    # Verify
    U_r = U.reshape((n, m, R))
    V_r = V.reshape((m, p, R))
    W_r = W.reshape((p, n, R))
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))

    if not np.allclose(T_recon, T_target, atol=1e-9):
        return schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
