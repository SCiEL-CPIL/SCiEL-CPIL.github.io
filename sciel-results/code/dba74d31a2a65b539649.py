"""
Strategy: The 2x4x5 matrix multiplication tensor T_{i,j,k} = 1 for all valid indices.
We decompose it as T = sum_{j=0}^{3} (sum_i e_i) ⊗ e_j ⊗ (sum_k e_k), giving rank 4.
U[i*m+j, c] = 1 if c == j, else 0  (for all i)
V[j*p+k, c] = 1 if j == c, else 0  (for all k)
W[k*n+i, c] = 1 for all i, k, c
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140

    rank = m  # 4
    U = np.zeros((n * m, rank))  # 8 x 4
    V = np.zeros((m * p, rank))  # 20 x 4
    W = np.zeros((p * n, rank))  # 10 x 4

    # U[i*m + j, c] = 1 if c == j
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1.0

    # V[j*p + k, c] = 1 if j == c
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1.0

    # W[k*n + i, c] = 1 for all i, k, c
    for k in range(p):
        for i in range(n):
            for c in range(rank):
                W[k * n + i, c] = 1.0

    # Verify the decomposition
    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check invalid positions give 0
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if idx_u == i * m + j and idx_v == j * p + k and idx_w == k * n + i:
                                    valid = True
                                    break
                            if valid:
                                break
                        if valid:
                            break
                    if valid:
                        continue
                    val = 0.0
                    for c in range(r):
                        val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                    if abs(val) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_s = n * m * p
    U_s = np.zeros((n * m, rank_s))
    V_s = np.zeros((m * p, rank_s))
    W_s = np.zeros((p * n, rank_s))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, c] = 1.0
                V_s[j * p + k, c] = 1.0
                W_s[k * n + i, c] = 1.0
                c += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
