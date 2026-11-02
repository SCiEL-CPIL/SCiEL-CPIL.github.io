import numpy as np

def construct():
    """
    Rank-56 decomposition of the 2x4 * 4x8 matrix multiplication tensor
    via Strassen applied to each of the 8 non-overlapping 2x2x2x2 sub-blocks.
    """
    n, m, p = 2, 4, 8

    R = 56  # 8 blocks * 7 products per block
    U = np.zeros((n * m, R), dtype=np.float64)  # (8, 56)
    V = np.zeros((m * p, R), dtype=np.float64)  # (32, 56)
    W = np.zeros((p * n, R), dtype=np.float64)  # (16, 56)

    # For each of the 8 (u, v) blocks, apply Strassen's 7-product scheme.
    for u in range(m // 2):          # 2 blocks along middle dim (j)
        for v in range(p // 2):      # 4 blocks along last dim (k)
            col0 = 7 * (2 * v + u)

            # A row indices: A[i, j] -> i*m + j
            a = np.array([[0 * m + 2 * u + 0, 0 * m + 2 * u + 1],
                          [1 * m + 2 * u + 0, 1 * m + 2 * u + 1]])

            # B row indices: B[j, k] -> j*p + k
            b = np.array([[2 * u * p + 2 * v + 0, 2 * u * p + 2 * v + 1],
                          [(2 * u + 1) * p + 2 * v + 0, (2 * u + 1) * p + 2 * v + 1]])

            # C/W indices: C[i, k] -> k*n + i  (trace convention)
            c = np.array([(2 * v + 0) * n + 0, (2 * v + 0) * n + 1],
                         dtype=int)
            c2 = np.array([(2 * v + 1) * n + 0, (2 * v + 1) * n + 1],
                          dtype=int)
            # c[0]=C(0,2v), c[1]=C(1,2v), c2[0]=C(0,2v+1), c2[1]=C(1,2v+1)

            A11, A12 = a[0, 0], a[0, 1]
            A21, A22 = a[1, 0], a[1, 1]
            B11, B12 = b[0, 0], b[0, 1]
            B21, B22 = b[1, 0], b[1, 1]
            C11, C21 = c[0], c[1]
            C12, C22 = c2[0], c2[1]

            # M1 = (A11 + A22)(B11 + B22)
            #     C11 += M1; C22 += M1
            col = col0 + 0
            U[A11, col] += 1.0
            U[A22, col] += 1.0
            V[B11, col] += 1.0
            V[B22, col] += 1.0
            W[C11, col] += 1.0
            W[C22, col] += 1.0

            # M2 = (A21 + A22)B11
            #     C21 += M2; C22 -= M2
            col = col0 + 1
            U[A21, col] += 1.0
            U[A22, col] += 1.0
            V[B11, col] += 1.0
            W[C21, col] += 1.0
            W[C22, col] -= 1.0

            # M3 = A11(B12 - B22)
            #     C12 += M3; C22 += M3
            col = col0 + 2
            U[A11, col] += 1.0
            V[B12, col] += 1.0
            V[B22, col] -= 1.0
            W[C12, col] += 1.0
            W[C22, col] += 1.0

            # M4 = A22(B21 - B11)
            #     C11 += M4; C21 += M4
            col = col0 + 3
            U[A22, col] += 1.0
            V[B21, col] += 1.0
            V[B11, col] -= 1.0
            W[C11, col] += 1.0
            W[C21, col] += 1.0

            # M5 = (A11 + A12)B22
            #     C11 -= M5; C12 += M5
            col = col0 + 4
            U[A11, col] += 1.0
            U[A12, col] += 1.0
            V[B22, col] += 1.0
            W[C11, col] -= 1.0
            W[C12, col] += 1.0

            # M6 = (A21 - A11)(B11 + B12)
            #     C22 += M6
            col = col0 + 5
            U[A21, col] += 1.0
            U[A11, col] -= 1.0
            V[B11, col] += 1.0
            V[B12, col] += 1.0
            W[C22, col] += 1.0

            # M7 = (A12 - A22)(B21 + B22)
            #     C11 += M7
            col = col0 + 6
            U[A12, col] += 1.0
            U[A22, col] -= 1.0
            V[B21, col] += 1.0
            V[B22, col] += 1.0
            W[C11, col] += 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
