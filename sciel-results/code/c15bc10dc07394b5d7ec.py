"""Combined program: Strassen-based rank-55 decomposition with verification and fallback."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a matrix multiplication tensor decomposition for 2x3 * 3x10.
    Uses Strassen for 2x2 blocks (rank 7) + schoolbook for cross terms (rank 4)
    per 2-column block of the output. 5 blocks total → rank 55.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    deadline = time.time() + 115  # safety margin

    # Strategy:
    # For each of 5 column-pairs (k0, k1) of B (and C):
    #   A_L = A[:, 0:2] (2x2), A_R = A[:, 2] (2x1)
    #   B_L = B[0:2, k0:k1+1] (2x2), B_R = B[2, k0:k1+1] (1x2)
    #   C_block = A_L @ B_L + A_R @ B_R  (2x2)
    #   Strassen on A_L @ B_L: rank 7
    #   Schoolbook on A_R @ B_R: rank 4
    #   Total per block: 11
    #   Total: 5 * 11 = 55

    rank = 55
    U = np.zeros((n * m, rank))  # (6, 55)
    V = np.zeros((m * p, rank))  # (30, 55)
    W = np.zeros((p * n, rank))  # (20, 55)

    col = 0

    # Indices for A (2x3, row-major: i*m + j)
    a00 = 0 * m + 0  # 0
    a01 = 0 * m + 1  # 1
    a10 = 1 * m + 0  # 3
    a11 = 1 * m + 1  # 4
    a02 = 0 * m + 2  # 2
    a12 = 1 * m + 2  # 5

    for b in range(5):
        if time.time() > deadline:
            break
        k0 = 2 * b
        k1 = 2 * b + 1

        # B indices (3x10, row-major: j*p + k)
        b00 = 0 * p + k0  # B[0, k0]
        b01 = 0 * p + k1  # B[0, k1]
        b10 = 1 * p + k0  # B[1, k0]
        b11 = 1 * p + k1  # B[1, k1]
        b20 = 2 * p + k0  # B[2, k0]
        b21 = 2 * p + k1  # B[2, k1]

        # W/C indices (2x10, trace convention: k*n + i)
        c00 = k0 * n + 0  # C[0, k0]
        c01 = k1 * n + 0  # C[0, k1]
        c10 = k0 * n + 1  # C[1, k0]
        c11 = k1 * n + 1  # C[1, k1]

        # Strassen terms for A_L @ B_L (2x2 @ 2x2)
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1

        # M2 = (A10 + A11)B00
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] -= 1
        col += 1

        # M3 = A00(B01 - B11)
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] -= 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1

        # M4 = A11(B10 - B00)
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] -= 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1

        # M5 = (A00 + A01)B11
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1
        W[c01, col] += 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1
        U[a00, col] -= 1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1

        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1
        U[a11, col] -= 1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1

        # Schoolbook terms for A_R @ B_R (2x1 @ 1x2)
        # P1 = A02 * B20 -> C[0, k0]
        U[a02, col] += 1
        V[b20, col] += 1
        W[c00, col] += 1
        col += 1

        # P2 = A02 * B21 -> C[0, k1]
        U[a02, col] += 1
        V[b21, col] += 1
        W[c01, col] += 1
        col += 1

        # P3 = A12 * B20 -> C[1, k0]
        U[a12, col] += 1
        V[b20, col] += 1
        W[c10, col] += 1
        col += 1

        # P4 = A12 * B21 -> C[1, k1]
        U[a12, col] += 1
        V[b21, col] += 1
        W[c11, col] += 1
        col += 1

    # Verify the decomposition
    T_from_decomp = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_from_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]

    T_expected = np.ones((n, m, p))

    if not np.allclose(T_from_decomp, T_expected, atol=1e-10, rtol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
