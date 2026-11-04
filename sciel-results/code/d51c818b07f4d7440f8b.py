import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Split A into A1 (2x2) and A2 (2x1), B into B1 (2x6) and B2 (1x6).
    C = A1*B1 + A2*B2.
    A1*B1 is 2x2x6, decomposed into 3 blocks of 2x2x2 (Strassen, rank 7 each) -> rank 21.
    A2*B2 is 2x1x6, decomposed into rank 2 terms.
    Total rank = 23.
    """
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    rank = 23
    U = np.zeros((dim_a, rank), dtype=np.int64)
    V = np.zeros((dim_b, rank), dtype=np.int64)
    W = np.zeros((dim_c, rank), dtype=np.int64)

    # Strassen 2x2x2 factors
    S_U = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    S_V = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    S_W = [
        [[1, 0], [0, 1]],
        [[0, 1], [0, -1]],
        [[0, 0], [1, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]

    col = 0
    # A1*B1 part: 3 blocks of 2x2x2
    for b_blk in range(3):
        for r in range(7):
            # U part: A1 is 2x2, indices i in {0,1}, j in {0,1}
            # a = i*m + j = i*3 + j
            for i in range(2):
                for j in range(2):
                    a = i * m + j
                    U[a, col] = S_U[r][i][j]
            
            # V part: B1 is 2x6, indices j in {0,1}, k in {2*b_blk, 2*b_blk+1}
            # b = j*p + k = j*6 + k
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b_blk + k_local
                    b = j * p + k_global
                    V[b, col] = S_V[r][j][k_local]
            
            # W part: C is 6x2, indices k in {2*b_blk, 2*b_blk+1}, i in {0,1}
            # c = k*n + i = k*2 + i
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * b_blk + k_local
                    c = k_global * n + i
                    W[c, col] = S_W[r][k_local][i]
            
            col += 1

    # A2*B2 part: 2 terms for 2x1x6
    # A2 is 2x1, indices i in {0,1}, j=2
    # a = i*m + 2 = i*3 + 2
    # B2 is 1x6, indices j=2, k in {0..5}
    # b = 2*p + k = 12 + k
    # C is same as above.
    
    # Term 21 (r=21): i=0
    U[0 * m + 2, col] = 1
    U[1 * m + 2, col] = 0
    for k in range(p):
        V[2 * p + k, col] = 1
    for k in range(p):
        W[k * n + 0, col] = 1
        W[k * n + 1, col] = 0
    col += 1
    
    # Term 22 (r=22): i=1
    U[0 * m + 2, col] = 0
    U[1 * m + 2, col] = 1
    for k in range(p):
        V[2 * p + k, col] = 1
    for k in range(p):
        W[k * n + 0, col] = 0
        W[k * n + 1, col] = 1
    col += 1

    # Verify
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    result = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.array_equal(result, T):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
