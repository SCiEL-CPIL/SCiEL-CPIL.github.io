import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Block-parallel composition:
    #   Block A: 2x2x2 core  -> Strassen, rank 7
    #   Block B: 2x1x2 strip -> schoolbook, rank 4
    #   Block C: 2x3x1 strip -> schoolbook, rank 6
    # Total rank = 17

    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Index conventions:
    #   A[i,j]  -> U index  i*m + j   (i in 0..1, j in 0..2)
    #   B[j,k]  -> V index  j*p + k   (j in 0..2, k in 0..2)
    #   C[k,i]  -> W index  k*n + i   (k in 0..2, i in 0..1)  [trace convention]

    # ---- Strassen for the 2x2x2 block (rows/cols 0,1) ----
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1.0; U[4, 0] = 1.0
    V[0, 0] = 1.0; V[4, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0

    # r=1: (A10+A11)B00
    U[3, 1] = 1.0; U[4, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0; W[3, 1] = -1.0

    # r=2: A00(B01-B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[4, 2] = -1.0
    W[2, 2] = 1.0; W[3, 2] = 1.0

    # r=3: A11(B10-B00)
    U[4, 3] = 1.0
    V[3, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[1, 3] = 1.0

    # r=4: (A00+A01)B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[4, 4] = 1.0
    W[0, 4] = -1.0; W[2, 4] = 1.0

    # r=5: (A10-A00)(B00+B01)
    U[3, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[3, 5] = 1.0

    # r=6: (A01-A11)(B10+B11)
    U[1, 6] = 1.0; U[4, 6] = -1.0
    V[3, 6] = 1.0; V[4, 6] = 1.0
    W[0, 6] = 1.0

    # ---- Schoolbook for the 2x1x2 block: A[:,2] * B[2,:] ----
    # r=7: A02 * B20
    U[2, 7] = 1.0
    V[6, 7] = 1.0
    W[0, 7] = 1.0

    # r=8: A02 * B21
    U[2, 8] = 1.0
    V[7, 8] = 1.0
    W[2, 8] = 1.0

    # r=9: A12 * B20
    U[5, 9] = 1.0
    V[6, 9] = 1.0
    W[1, 9] = 1.0

    # r=10: A12 * B21
    U[5, 10] = 1.0
    V[7, 10] = 1.0
    W[3, 10] = 1.0

    # ---- Schoolbook for the 2x3x1 block: A[:,:] * B[:,2] ----
    # r=11: A00 * B02
    U[0, 11] = 1.0
    V[2, 11] = 1.0
    W[4, 11] = 1.0

    # r=12: A01 * B12
    U[1, 12] = 1.0
    V[5, 12] = 1.0
    W[4, 12] = 1.0

    # r=13: A02 * B22
    U[2, 13] = 1.0
    V[8, 13] = 1.0
    W[4, 13] = 1.0

    # r=14: A10 * B02
    U[3, 14] = 1.0
    V[2, 14] = 1.0
    W[5, 14] = 1.0

    # r=15: A11 * B12
    U[4, 15] = 1.0
    V[5, 15] = 1.0
    W[5, 15] = 1.0

    # r=16: A12 * B22
    U[5, 16] = 1.0
    V[8, 16] = 1.0
    W[5, 16] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
