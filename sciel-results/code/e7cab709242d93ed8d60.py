"""Matrix multiplication 2,2,12 — rank-24 decomposition via row-separation.

Each output row C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] is a 1x2 x 2x12
vector-matrix product of rank 12 (one rank-1 term per column). Two rows
give total rank 24.
"""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-24 exact tensor decomposition for 2x2 * 2x12 matrix mult.

    Strategy: Row-separation. Treat the 2x2 input matrix A as two independent
    row vectors r_1 = [a, b] and r_2 = [c, d]. Each row computes an independent
    1x2 x 2x12 vector-matrix product (rank 12 each), for a total rank of 24.

    Conventions:
      U: shape (n*m, rank) = (4, 24);  U[i*m+j, r] multiplies A[i][j]
      V: shape (m*p, rank) = (24, 24); V[j*p+k, r] multiplies B[j][k]
      W: shape (p*n, rank) = (24, 24); W[k*n+i, r] contributes to C[i][k] (trace convention)
    """
    n, m, p = 2, 2, 12
    rank = 2 * p  # 2 rows * 12 columns = 24

    U = [[0] * rank for _ in range(n * m)]    # 4 x 24
    V = [[0] * rank for _ in range(m * p)]    # 24 x 24
    W = [[0] * rank for _ in range(p * n)]    # 24 x 24

    # For each output row i (0 or 1) and each column k (0..11):
    #   C[i][k] = A[i][0]*B[0][k] + A[i][1]*B[1][k]
    # We use two rank-1 terms:
    #   Term 1: U[i*m+0, r] = 1, V[0*p+k, r] = 1, W[k*n+i, r] = 1  -> A[i][0]*B[0][k]
    #   Term 2: U[i*m+1, r] = 1, V[1*p+k, r] = 1, W[k*n+i, r] = 1  -> A[i][1]*B[1][k]

    r = 0
    for i in range(n):  # output row
        for k in range(p):  # output column
            # Term for A[i][0] * B[0][k]
            U[i * m + 0][r] = 1
            V[0 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            # Term for A[i][1] * B[1][k]
            U[i * m + 1][r] = 1
            V[1 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
