import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen rank-7 decomposition for 2x2 matrix multiplication
    # A flat: [A00, A01, A10, A11]  (index i*m+k, m=2)
    # B flat: [B00, B01, B10, B11]  (index k, within a 2x2 block)
    # C flat: [C00, C10, C01, C11]  (index j*n+i, n=2, within a 2-col block)
    #
    # The 7 terms (u, v, w) where u·v contributes to w positions:
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    start_time = time.time()
    deadline = start_time + 135.0

    col = 0
    for blk in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            # U: A indices [i*m + k] = [i*2 + k]
            U[0, col] = a_vec[0]  # A[0,0]
            U[1, col] = a_vec[1]  # A[0,1]
            U[2, col] = a_vec[2]  # A[1,0]
            U[3, col] = a_vec[3]  # A[1,1]

            # V: B indices [k*p + j], block covers cols 2*blk, 2*blk+1
            V[0 * p + 2 * blk, col] = b_vec[0]
            V[0 * p + 2 * blk + 1, col] = b_vec[1]
            V[1 * p + 2 * blk, col] = b_vec[2]
            V[1 * p + 2 * blk + 1, col] = b_vec[3]

            # W: C indices [j*n + i], j is col of C, i is row of C
            W[2 * blk * n + 0, col] = c_vec[0]
            W[2 * blk * n + 1, col] = c_vec[1]
            W[(2 * blk + 1) * n + 0, col] = c_vec[2]
            W[(2 * blk + 1) * n + 1, col] = c_vec[3]

            col += 1

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
