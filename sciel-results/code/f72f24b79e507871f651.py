"""Strassen-based tensor decomposition for 2x3x10 matrix multiplication."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # We decompose into:
    #   1) 2x2x10 sub-problem (rows 0-1 of A, cols 0-1 of B, all 10 rows of B)
    #   2) 2x1x10 sub-problem (rows 0-1 of A, col 2 of B, all 10 rows of B)
    # Using Strassen on 2x2 blocks for the first part.

    # Total rank: Strassen(2x2) per 2-column block + schoolbook for 2x1
    # 2x2x10: split into 5 blocks of 2 columns each, each uses Strassen (rank 7)
    #   => 5 * 7 = 35
    # 2x1x10: schoolbook, rank 2*1*10 = 20
    # Total rank = 35 + 20 = 55

    rank = 35 + 20
    U = [[0] * rank for _ in range(n * m)]       # 6 x 55
    V = [[0] * rank for _ in range(m * p)]       # 30 x 55
    W = [[0] * rank for _ in range(p * n)]       # 20 x 55

    # --- Part 1: 2x2x10 using Strassen on each 2-column block ---
    # A has shape (2,2) for this part: rows i=0,1; cols j=0,1
    # B has shape (2,10): rows j=0,1; cols k=0..9
    # We split B's 10 columns into 5 blocks of 2: k=2b, 2b+1

    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        # 7 Strassen terms
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0 * 2 + 0][col] += 1   # A[0,0]
        U[1 * 2 + 1][col] += 1   # A[1,1]
        V[0 * 10 + k0][col] += 1  # B[0, k0]
        V[1 * 10 + k1][col] += 1  # B[1, k1]
        # C[0,k0] += M1, C[0,k1] += M1, C[1,k0] -= M1, C[1,k1] += M1
        # W index: k*n+i = k*2+i
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] -= 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2 = (A10 + A11) * (B0k0)
        U[1 * 2 + 0][col] += 1   # A[1,0]
        U[1 * 2 + 1][col] += 1   # A[1,1]
        V[0 * 10 + k0][col] += 1  # B[0, k0]
        # C[1,k0] += M2, C[1,k1] -= M2
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3 = A00 * (B1k0 + B1k1)
        U[0 * 2 + 0][col] += 1   # A[0,0]
        V[1 * 10 + k0][col] += 1  # B[1, k0]
        V[1 * 10 + k1][col] += 1  # B[1, k1]
        # C[0,k1] += M3, C[1,k1] += M3
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4 = A11 * (B1k1 - B0k1)
        U[1 * 2 + 1][col] += 1   # A[1,1]
        V[1 * 10 + k1][col] += 1  # B[1, k1]
        V[0 * 10 + k1][col] -= 1  # B[0, k1]
        # C[0,k0] += M4, C[1,k0] += M4
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5 = (A00 + A01) * (B1k1)
        U[0 * 2 + 0][col] += 1   # A[0,0]
        U[0 * 2 + 1][col] += 1   # A[0,1]
        V[1 * 10 + k1][col] += 1  # B[1, k1]
        # C[0,k0] -= M5, C[0,k1] += M5
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[1 * 2 + 0][col] += 1   # A[1,0]
        U[0 * 2 + 0][col] -= 1   # A[0,0]
        V[0 * 10 + k0][col] += 1  # B[0, k0]
        V[0 * 10 + k1][col] += 1  # B[0, k1]
        # C[1,k1] += M6
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[0 * 2 + 1][col] += 1   # A[0,1]
        U[1 * 2 + 1][col] -= 1   # A[1,1]
        V[1 * 10 + k0][col] += 1  # B[1, k0]
        V[1 * 10 + k1][col] += 1  # B[1, k1]
        # C[0,k0] += M7
        W[k0 * 2 + 0][col] += 1
        col += 1

    # --- Part 2: 2x1x10 schoolbook (col j=2 of B) ---
    for i in range(n):
        for k in range(p):
            U[i * 2 + 2][col] = 1   # A[i, 2]
            V[2 * 10 + k][col] = 1  # B[2, k]
            W[k * 2 + i][col] = 1   # C[i, k]
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
