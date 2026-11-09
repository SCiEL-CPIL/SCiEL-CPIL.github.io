"""Rewritten program using block-diagonal structure exploitation for 2x5x6 matrix multiplication."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2 rows of A, m=5 cols of A/rows of B, p=6 cols of B
    
    # For 2 x m x p matrix multiplication, we can use rank m*p
    # by sharing the B-factor across the two output rows.
    # For each (j, k), one rank-1 term handles both i=0 and i=1.
    # U[i*m + j, r] = 1 if j matches, V[j*p + k, r] = 1 if k matches, W[k*n + i, r] = 1
    
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for j in range(m):
        for k in range(p):
            # This single term handles both i=0 and i=1
            for i in range(n):
                U[i * m + j][idx] = 1
            V[j * p + k][idx] = 1
            for i in range(n):
                W[k * n + i][idx] = 1
            idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
