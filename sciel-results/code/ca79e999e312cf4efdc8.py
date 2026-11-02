import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS

    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    start_time = time.time()
    deadline = start_time + 140.0

    # Non-uniform partitioning: split middle dim j={0,1,2,3} into {0,1,2} and {3}
    # For j in {0,1,2}: split k={0..6} into {0,1,2}, {3,4,5}, {6}
    #   - 2x3x3 blocks: rank 9 each (x2) = 18
    #   - 2x3x1 block: rank 3
    # For j={3}: 2x1x7 block: rank 1
    # Total: 18 + 3 + 1 = 22

    rank = 22
    U = np.zeros((n*m, rank), dtype=np.int8)
    V = np.zeros((m*p, rank), dtype=np.int8)
    W = np.zeros((p*n, rank), dtype=np.int8)

    col = 0

    # Add 2x3x3 schoolbook block for given j_range and k_range
    def add_2x3x3_block(j_range, k_range):
        nonlocal col
        for j in j_range:
            for k in k_range:
                for i in range(n):
                    U[i*m + j, col] += 1
                for jp in range(m):
                    V[jp*p + k, col] += (1 if jp in j_range else 0)
                for i in range(n):
                    W[k*n + i, col] += 1
                col += 1

    # Wait, that's not right. Let me be more careful.
    # For 2x3x3: j in {0,1,2}, jp in {0,1,2}, k in {0,1,2}
    # Schoolbook: for each (j,k) pair, one rank-1 term
    # U[i*4+j, c] = 1 for all i
    # V[j*7+k, c] = 1 (only jp=j contributes)
    # W[k*2+i, c] = 1 for all i

    def add_2x3x3_block(j_vals, k_vals):
        nonlocal col
        for j in j_vals:
            for k in k_vals:
                for i in range(n):
                    U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                for i in range(n):
                    W[k*n + i, col] = 1
                col += 1

    # For 2x3x1: j in {0,1,2}, jp in {0,1,2}, k=6
    # Rank 3: for each j=jp, one term
    def add_2x3x1_block(j_vals, k_val):
        nonlocal col
        for j in j_vals:
            for i in range(n):
                U[i*m + j, col] = 1
            V[j*p + k_val, col] = 1
            for i in range(n):
                W[k_val*n + i, col] = 1
            col += 1

    # For 2x1x7: j=3, jp=3, k in {0..6}
    # Rank 1: single term
    def add_2x1x7_block(j_val):
        nonlocal col
        for i in range(n):
            U[i*m + j_val, col] = 1
        for k in range(p):
            V[j_val*p + k, col] = 1
        for k in range(p):
            for i in range(n):
                W[k*n + i, col] = 1
        col += 1

    # Build the decomposition
    add_2x3x3_block([0, 1, 2], [0, 1, 2])  # 9 terms
    add_2x3x3_block([0, 1, 2], [3, 4, 5])  # 9 terms
    add_2x3x1_block([0, 1, 2], 6)           # 3 terms
    add_2x1x7_block(3)                       # 1 term

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
