import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # We aim for rank 28 using Strassen for 2x2 blocks + schoolbook for the rest
    # 2x3 * 3x5
    # Split into:
    #   A_left (2x2) * B_left (2x2) -> C_left (2x2)  [Strassen, rank 7]
    #   A_left (2x2) * B_mid  (2x2) -> C_mid  (2x2)  [Strassen, rank 7]
    #   A_left (2x2) * B_right(2x1) -> C_right(2x1)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_left (2x2) -> C_left2(2x2)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_mid  (2x2) -> C_mid2 (2x2)  [Schoolbook, rank 4]
    #   A_right(2x1) * B_right(2x1) -> C_right2(2x1) [Schoolbook, rank 2]
    # Total: 7+7+4+4+4+2 = 28

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1

        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1

        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1

        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # 1. A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)

    # 2. A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)

    # 3. A[:, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    add_schoolbook([0, 1], [0, 1], [4])

    # 4. A[:, 2:3] * B[2:3, 0:2] -> C[0:2, 0:2]
    add_schoolbook([0, 1], [2], [0, 1])

    # 5. A[:, 2:3] * B[2:3, 2:4] -> C[0:2, 2:4]
    add_schoolbook([0, 1], [2], [2, 3])

    # 6. A[:, 2:3] * B[2:3, 4:5] -> C[0:2, 4:5]
    add_schoolbook([0, 1], [2], [4])

    # Verify exactness
    res = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]

    if not np.allclose(res, 1.0, atol=1e-9):
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
