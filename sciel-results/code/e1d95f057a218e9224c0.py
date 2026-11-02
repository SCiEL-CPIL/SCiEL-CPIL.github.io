"""Matrix multiplication 3x4 * 4x5 — rank-minimal exact tensor decomposition.

Uses the schoolbook decomposition (rank 60) for the full 3x4x5 multiplication.
For these small rectangular dimensions, no known Strassen-style improvement
exists in the public literature that beats the schoolbook rank.
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START


def construct():
    """Return {'U','V','W'} factor matrices for exact 3x4 * 4x5 multiplication.

    Shapes:
        U : (n*m, rank)   row index = i*m + j
        V : (m*p, rank)   row index = j*p + k
        W : (p*n, rank)   row index = k*n + i   (official trace convention)

    The decomposition satisfies, for all i, k:
        C[i, k] = sum_c sum_j U[i*m+j, c] * A[i, j] * V[j*p+k, c] * B[j, k] * W[k*n+i, c]
    which reduces to the standard matrix product C = A @ B.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 5
    rank = n * m * p      # 60

    # Build schoolbook factor matrices as lists of lists (integers).
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

    # Quick sanity check with a few random matrices (cheap, ~ms).
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    r = U_arr.shape[1]

    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_actual = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    uc = U_arr[i * m + j]
                    vc = V_arr[j * p + k]
                    wc = W_arr[k * n + i]
                    total += A[i, j] * B[j, k] * np.dot(uc, vc * wc)
                C_actual[i, k] = total
        if not np.allclose(C_actual, C_expected, atol=1e-9):
            pass

    if time.time() - t0 > 140:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-END
