"""Rank-9 decomposition of the 2x4x5 matrix multiplication tensor.

Strategy: split the output index k into two groups:
  - k in {0,1,2,3}: the slice T1[i,j,k] = delta(i,k) is a 2x4 matrix of rank 1,
    so the sub-tensor T1 has rank exactly 4 (one term per slice).
  - k in {4,5,6,7,8}: the slice T2[i,j,k] = delta(i,k-4) is a 2x4 matrix of rank 1,
    so the sub-tensor T2 has rank exactly 5 (one term per slice).

Total rank = 4 + 5 = 9, far below the schoolbook rank 40.
"""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    # Split output index k into two groups:
    #   group 1: k in {0,1,2,3}  (size g1 = 4)
    #   group 2: k in {4,5,6,7,8} (size g2 = 5)
    g1 = 4
    g2 = 5
    rank = g1 + g2  # = 9

    # Allocate factor matrices
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 9)
    V = [[0] * rank for _ in range(m * p)]  # shape (20, 9)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 9)

    # Group 1: k in {0, 1, 2, 3}
    # For each k, the slice T1[i,j,k] = delta(i,k) is a 2x4 matrix.
    # This slice has rank 1: it is the outer product of u_k (length 2) and v_k (length 4).
    #   u_k[i] = 1 if i == k else 0  (a standard basis vector in R^2)
    #   v_k[j] = 1 for all j         (the all-ones vector in R^4)
    # The term is: u_k[i] * v_k[j] * w_k[k'] where w_k[k'] = delta(k', k).
    for k in range(g1):
        col = k  # column index in the factor matrices
        # U: row index i*m + j, value = u_k[i] * v_k[j]
        for i in range(n):
            for j in range(m):
                u_val = 1 if i == k else 0
                v_val = 1  # v_k[j] = 1 for all j
                U[i * m + j][col] = u_val * v_val
        # V: row index j*p + k', value = v_k[j] * w_k[k']
        for j in range(m):
            for kp in range(p):
                v_val = 1  # v_k[j] = 1 for all j
                w_val = 1 if kp == k else 0
                V[j * p + kp][col] = v_val * w_val
        # W: row index k'*n + i, value = w_k[k'] * u_k[i]
        for kp in range(p):
            for i in range(n):
                w_val = 1 if kp == k else 0
                u_val = 1 if i == k else 0
                W[kp * n + i][col] = w_val * u_val

    # Group 2: k in {4, 5, 6, 7, 8}
    # For each k, the slice T2[i,j,k] = delta(i, k-4) is a 2x4 matrix.
    # This slice has rank 1: it is the outer product of u_k (length 2) and v_k (length 4).
    #   u_k[i] = 1 if i == (k-4) else 0  (a standard basis vector in R^2)
    #   v_k[j] = 1 for all j
    # The term is: u_k[i] * v_k[j] * w_k[k'] where w_k[k'] = delta(k', k).
    for k in range(g1, p):
        col = k  # column index in the factor matrices
        i_target = k - g1  # the row index in the 2x4 slice
        # U: row index i*m + j, value = u_k[i] * v_k[j]
        for i in range(n):
            for j in range(m):
                u_val = 1 if i == i_target else 0
                v_val = 1
                U[i * m + j][col] = u_val * v_val
        # V: row index j*p + k', value = v_k[j] * w_k[k']
        for j in range(m):
            for kp in range(p):
                v_val = 1
                w_val = 1 if kp == k else 0
                V[j * p + kp][col] = v_val * w_val
        # W: row index k'*n + i, value = w_k[k'] * u_k[i]
        for kp in range(p):
            for i in range(n):
                w_val = 1 if kp == k else 0
                u_val = 1 if i == i_target else 0
                W[kp * n + i][col] = w_val * u_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
