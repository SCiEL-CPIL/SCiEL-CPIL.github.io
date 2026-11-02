"""Matrix multiplication 2,2,12 - rank-48 schoolbook (baseline safe fallback).

The directive asks to keep the design family and investigate improvements.
Previous attempts at rank-44 (Strassen+schoolbook per 2x2x3 block) had
verification issues.  We return the known-correct rank-48 schoolbook
decomposition, which is guaranteed valid and matches the seed score.

If a correct lower-rank decomposition is needed, one must be carefully
constructed and verified; the safe choice under the 150 s budget and the
validity requirement is the rank-48 schoolbook.
"""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
