import numpy as np
import time

DIMENSIONS = [4, 4, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 120
    n, m, p = DIMENSIONS  # n=4, m=4, p=9

    # Correct rank-114 block scheme:
    #   A*B1 (B1 = B[:,0:4]) via Strassen 4x4x4  -> rank 49
    #   A*B2 (B2 = B[:,4:8]) via Strassen 4x4x4  -> rank 49
    #   A*B3 (B3 = B[:,8:9]) via schoolbook 4x4x1 -> rank 16
    U44, V44, W44 = _build_strassen_4x4x4()
    U41, V41, W41 = _build_schoolbook_4x4x1()

    r1, r2, r3 = 49, 49, 16
    total_rank = r1 + r2 + r3

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    # Part 1: A * B1 (cols 0..3)
    U[:, 0:r1] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:r1] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:r1] = W44[j * 4 + i, :]

    # Part 2: A * B2 (cols 4..7)
    U[:, r1:r1 + r2] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), r1:r1 + r2] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, r1:r1 + r2] = W44[j * 4 + i, :]

    # Part 3: A * B3 (col 8)
    U[:, r1 + r2:r1 + r2 + r3] = U41
    for k in range(4):
        V[k * 9 + 8, r1 + r2:r1 + r2 + r3] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, r1 + r2:r1 + r2 + r3] = W41[i, :]

    # Fast vectorized self-check against the true product.
    if time.time() < deadline - 3:
        rng = np.random.default_rng(12345)
        A_test = rng.standard_normal((4, 4))
        B_test = rng.standard_normal((4, 9))
        C_expected = A_test @ B_test

        Uv = U.reshape(4, 4, total_rank)
        Vv = V.reshape(4, 9, total_rank)
        Wv = W.reshape(9, 4, total_rank)

        a_lin = np.einsum('ikr,ik->ir', Uv, A_test)
        b_lin = np.einsum('kjr,kj->jr', Vv, B_test)
        C_computed = np.einsum('ijr,ir,jr->ij', Wv, a_lin, b_lin)

        if not np.allclose(C_computed, C_expected, atol=1e-8):
            # Fallback: exact schoolbook (rank 144)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            idx = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k, idx] = 1
                        V[k * p + j, idx] = 1
                        W[j * n + i, idx] = 1
                        idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_strassen_4x4x4():
    """Rank-49 Strassen decomposition for 4x4 x 4x4 (Kronecker of 2x2 Strassen)."""
    U2, V2, W2 = _build_strassen_2x2x2()

    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    big_mults = [
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]

    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi, bk), sign_A in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * bi + i2
                        k = 2 * bk + k2
                        U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
            for (bk, bj), sign_B in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k = 2 * bk + k2
                        j = 2 * bj + j2
                        V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
            for (bj, bi), sign_C in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j = 2 * bj + j2
                        i = 2 * bi + i2
                        W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]

    return U, V, W


def _build_strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2 x 2x2. Trace convention W[j*2+i, r]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_schoolbook_4x4x1():
    """Rank-16 schoolbook decomposition for 4x4 x 4x1. Trace convention W[i, r]."""
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))

    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U[i * 4 + k, r] = 1
            V[k, r] = 1
            W[i, r] = 1

    return U, V, W
