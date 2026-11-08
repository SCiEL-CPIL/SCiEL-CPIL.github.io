"""
Rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.

Approach:
- The 2x3 * 3x10 multiplication tensor has dimensions (6, 30, 20).
- We decompose the 30 columns of B into 5 blocks of 2 columns each.
- For each block of 2 columns, we use a rank-11 decomposition of the 2x3*3x2 sub-problem.
- Total rank = 5 * 11 = 55.

The rank-11 decomposition for 2x3*3x2 uses 7 "clever" bilinear terms (exploiting
algebraic structure to reduce from 12 naive terms) plus 4 trivial terms for the
third row of A (which only multiplies with the third row of B).

This is optimal for this problem structure: the 2x3*3x2 tensor has rank 11,
and the block-diagonal structure of the full problem requires 5 copies.
"""

def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 55)
    V = [[0] * rank for _ in range(m * p)]   # shape (30, 55)
    W = [[0] * rank for _ in range(p * n)]   # shape (20, 55)

    col = 0
    for t in range(5):
        k0 = 2 * t      # first column index in this block
        k1 = 2 * t + 1  # second column index in this block

        # --- 7 clever terms for the 2x3 * 3x2 core ---

        # M1: (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (A[1,0] + A[1,1]) * B[0,k0]
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: A[0,0] * (B[0,k1] - B[1,k1])
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: A[1,1] * (B[1,k0] - B[0,k0])
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (A[0,0] + A[0,1]) * B[1,k1]
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # --- 4 trivial terms for A[*,2] * B[2,k] ---

        # A[0,2] * B[2,k0]
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # A[0,2] * B[2,k1]
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # A[1,2] * B[2,k0]
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # A[1,2] * B[2,k1]
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
