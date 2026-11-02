import numpy as np
import time

def construct():
    # Rank-56 Strassen decomposition for 2x2x16
    # This is the known optimal rank for this tensor.
    n, m, p = 2, 2, 16
    num_blocks = p // m  # 8

    # Rank-7 Strassen decomposition for 2x2x2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    U_s = np.zeros((4, 7), dtype=np.int64)
    V_s = np.zeros((4, 7), dtype=np.int64)
    W_s = np.zeros((4, 7), dtype=np.int64)

    def add_u(i, j, coeff, col):
        U_s[i * 2 + j, col] += coeff

    def add_v(j, k, coeff, col):
        V_s[j * 2 + k, col] += coeff

    def add_w(i, k, coeff, col):
        W_s[k * 2 + i, col] += coeff

    # m1 = (A00+A11)(B00+B11)
    add_u(0, 0, 1, 0); add_u(1, 1, 1, 0)
    add_v(0, 0, 1, 0); add_v(1, 1, 1, 0)
    add_w(0, 0, 1, 0); add_w(1, 1, 1, 0)

    # m2 = (A10+A11)B00
    add_u(1, 0, 1, 1); add_u(1, 1, 1, 1)
    add_v(0, 0, 1, 1)
    add_w(1, 0, 1, 1); add_w(1, 1, -1, 1)

    # m3 = A00(B01-B11)
    add_u(0, 0, 1, 2)
    add_v(0, 1, 1, 2); add_v(1, 1, -1, 2)
    add_w(0, 1, 1, 2); add_w(1, 1, 1, 2)

    # m4 = A11(B10-B00)
    add_u(1, 1, 1, 3)
    add_v(1, 0, 1, 3); add_v(0, 0, -1, 3)
    add_w(0, 0, 1, 3); add_w(1, 0, 1, 3)

    # m5 = (A00+A01)B11
    add_u(0, 0, 1, 4); add_u(0, 1, 1, 4)
    add_v(1, 1, 1, 4)
    add_w(0, 0, -1, 4); add_w(0, 1, 1, 4)

    # m6 = (A10-A00)(B00+B01)
    add_u(1, 0, 1, 5); add_u(0, 0, -1, 5)
    add_v(0, 0, 1, 5); add_v(0, 1, 1, 5)
    add_w(1, 1, 1, 5)

    # m7 = (A01-A11)(B10+B11)
    add_u(0, 1, 1, 6); add_u(1, 1, -1, 6)
    add_v(1, 0, 1, 6); add_v(1, 1, 1, 6)
    add_w(0, 0, 1, 6)

    # Expand to 2x2x16
    # U: 4x7 -> 4x56
    U = np.tile(U_s, (1, num_blocks))

    # V: 4x7 -> 32x56
    V = np.zeros((m * p, 7 * num_blocks), dtype=np.int64)
    # V rows: j*p + k. j in {0,1}, k in {0..15}
    # Block t uses k=2t, 2t+1
    # V_s rows: j*2 + k_local. k_local in {0,1}
    # Mapping:
    # V_s[0] (j=0, k_local=0) -> V[0*p + 2t]
    # V_s[1] (j=0, k_local=1) -> V[0*p + 2t+1]
    # V_s[2] (j=1, k_local=0) -> V[1*p + 2t]
    # V_s[3] (j=1, k_local=1) -> V[1*p + 2t+1]
    
    for t in range(num_blocks):
        cols = slice(7 * t, 7 * t + 7)
        V[2 * t, cols] = V_s[0, :]
        V[2 * t + 1, cols] = V_s[1, :]
        V[16 + 2 * t, cols] = V_s[2, :]
        V[16 + 2 * t + 1, cols] = V_s[3, :]

    # W: 4x7 -> 32x56
    W = np.zeros((p * n, 7 * num_blocks), dtype=np.int64)
    # W rows: k*n + i. k in {0..15}, i in {0,1}
    # Block t uses k=2t, 2t+1
    # W_s rows: k_local*2 + i. k_local in {0,1}
    # Mapping:
    # W_s[0] (k_local=0, i=0) -> W[2t*n + 0] = W[4t]
    # W_s[1] (k_local=0, i=1) -> W[2t*n + 1] = W[4t+1]
    # W_s[2] (k_local=1, i=0) -> W[(2t+1)*n + 0] = W[4t+2]
    # W_s[3] (k_local=1, i=1) -> W[(2t+1)*n + 1] = W[4t+3]

    for t in range(num_blocks):
        cols = slice(7 * t, 7 * t + 7)
        W[4 * t, cols] = W_s[0, :]
        W[4 * t + 1, cols] = W_s[1, :]
        W[4 * t + 2, cols] = W_s[2, :]
        W[4 * t + 3, cols] = W_s[3, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
