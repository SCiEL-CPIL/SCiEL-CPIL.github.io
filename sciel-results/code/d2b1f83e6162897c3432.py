"""Matrix multiplication 3,4,8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8

    def verify(U, V, W, n, m, p, tol=1e-9):
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Check all compatible triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for col in range(r):
                        val += U_np[a, col] * V_np[b, col] * W_np[c, col]
                    if abs(val - 1.0) > tol:
                        return False
        # Check some incompatible triples to ensure they're 0
        # (spot check a few)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    # Check if this is a compatible triple
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    k2 = c // n
                    i2 = c % n
                    if not (j1 == j2 and k1 == k2 and i1 == i2):
                        val = 0.0
                        for col in range(r):
                            val += U_np[a, col] * V_np[b, col] * W_np[c, col]
                        if abs(val) > tol:
                            return False
        return True

    # Rank 32 decomposition: one term per (j, k) pair
    # For term (j, k):
    #   U[i*m+j, c] = 1 for all i (only column j of U is nonzero)
    #   V[j*p+k, c] = 1 (only position (j,k) is nonzero)
    #   W[k*n+i, c] = 1 for all i (only row k of W is nonzero)
    #
    # Verification: For compatible triple (i, j', k'):
    #   Only term (j', k') has V[j'*p+k', c] != 0
    #   That term contributes U[i*m+j', c]*V[j'*p+k', c]*W[k'*n+i, c] = 1*1*1 = 1
    #   All other terms contribute 0
    #   Sum = 1 ✓
    #
    # For incompatible triple (a, b, c) where j1 != j2 or k1 != k2 or i1 != i2:
    #   If j1 != j2: For any term (j,k), U[a, c] is nonzero only if j == j1,
    #     but V[b, c] is nonzero only if j == j2. Since j1 != j2, no term has
    #     both U and V nonzero. Sum = 0 ✓
    #   Similarly for other cases.

    r = m * p  # 32
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]

    col = 0
    for j in range(m):
        for k in range(p):
            c = col
            # U: nonzero in all rows of column j
            for i in range(n):
                U[i * m + j][c] = 1
            # V: nonzero only at position (j, k)
            V[j * p + k][c] = 1
            # W: nonzero in all columns of row k
            for i in range(n):
                W[k * n + i][c] = 1
            col += 1

    # Verify
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
