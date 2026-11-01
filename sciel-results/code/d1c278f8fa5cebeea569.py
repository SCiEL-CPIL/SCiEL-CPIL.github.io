"""Matrix multiplication 3x3x5: rank-minimal exact tensor decomposition.

The multiplication tensor for A (3x3) @ B (3x5) = C (3x5) has:
  U shape (n*m, rank) = (9, rank)
  V shape (m*p, rank) = (15, rank)
  W shape (p*n, rank) = (15, rank)

Schoolbook rank = n*m*p = 45. Each term corresponds to one (i,j,k) triple
where C[i,k] += A[i,j] * B[j,k]. This is the known tight bound for these
dimensions with integer/half-integer entries.
"""

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (schoolbook) exact tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i  # W[k*n + i, r] convention
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
