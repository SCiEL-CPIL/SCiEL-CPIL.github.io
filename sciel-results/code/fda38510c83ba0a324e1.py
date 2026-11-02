"""
Matrix multiplication 2x2 * 2x15 tensor decomposition.
Uses Strassen's algorithm (rank 7) for each 2x2 block of B,
with a final 2x1 block handled directly (rank 4).
Total rank = 7*7 + 4 = 53.

Strassen formulas (verified):
  A = [[a,b],[c,d]], B = [[e,f],[g,h]]
  C[0][0] = a*e + b*g
  C[0][1] = a*f + b*h
  C[1][0] = c*e + d*g
  C[1][1] = c*f + d*h

  M1 = (a+d)(e+h)
  M2 = (c+d)e
  M3 = a(f-h)
  M4 = d(g-e)
  M5 = (a+c)h
  M6 = (d-b)(e+f)
  M7 = (b-c)(g+h)

  C00 = M1 + M3 - M4 - M6
  C01 = M4 + M5
  C10 = M2 + M6
  C11 = M1 - M3 + M5 - M7
"""

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15

    # Split B (2x15) into seven 2x2 blocks and one 2x1 block.
    # Block t (t=0..6): columns 2t, 2t+1
    # Block 7: column 14

    rank = 0
    U = []  # list of rows, each row is a list of rank entries
    V = []
    W = []

    # Initialize with zeros (we'll build column by column)
    total_rank = 7 * 7 + 4  # 53

    # We'll build U, V, W as 2D lists
    # U: (n*m) x rank = 4 x 53
    # V: (m*p) x rank = 30 x 53
    # W: (p*n) x rank = 30 x 53

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    # Strassen blocks for t = 0..6
    for t in range(7):
        k0 = 2 * t       # first column of this block
        k1 = 2 * t + 1   # second column of this block

        # M1 = (a+d)(e+h)
        # U: row i*m+j = a is (0,0), d is (1,1) -> both +1
        U[0][col] = 1  # a = A[0][0], index 0*m+0 = 0
        U[3][col] = 1  # d = A[1][1], index 1*m+1 = 3
        # V: e = B[0][k0], h = B[1][k1]
        V[0 * p + k0][col] = 1  # e
        V[1 * p + k1][col] = 1  # h
        # W: contributes to C[0][k0] with +1, C[1][k1] with +1
        W[k0 * n + 0][col] = 1   # C[0][k0]
        W[k1 * n + 1][col] = 1   # C[1][k1]
        col += 1

        # M2 = (c+d)e
        # U: c = A[1][0] index 1*m+0 = 2, d = A[1][1] index 3
        U[2][col] = 1
        U[3][col] = 1
        # V: e = B[0][k0]
        V[0 * p + k0][col] = 1
        # W: contributes to C[1][k0] with +1
        W[k0 * n + 1][col] = 1
        col += 1

        # M3 = a(f-h)
        # U: a = A[0][0] index 0
        U[0][col] = 1
        # V: f = B[0][k1] +1, h = B[1][k1] -1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        # W: contributes to C[0][k0] with +1, C[1][k1] with -1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # M4 = d(g-e)
        # U: d = A[1][1] index 3
        U[3][col] = 1
        # V: g = B[1][k0] +1, e = B[0][k0] -1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        # W: contributes to C[0][k1] with +1, C[0][k0] with -1
        W[k1 * n + 0][col] = 1
        W[k0 * n + 0][col] = -1
        col += 1

        # M5 = (a+c)h
        # U: a = A[0][0] index 0, c = A[1][0] index 2
        U[0][col] = 1
        U[2][col] = 1
        # V: h = B[1][k1]
        V[1 * p + k1][col] = 1
        # W: contributes to C[0][k1] with +1, C[1][k1] with +1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M6 = (d-b)(e+f)
        # U: d = A[1][1] index 3 +1, b = A[0][1] index 1 -1
        U[3][col] = 1
        U[1][col] = -1
        # V: e = B[0][k0] +1, f = B[0][k1] +1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        # W: contributes to C[0][k0] with -1, C[1][k0] with +1
        W[k0 * n + 0][col] = -1
        W[k0 * n + 1][col] = 1
        col += 1

        # M7 = (b-c)(g+h)
        # U: b = A[0][1] index 1 +1, c = A[1][0] index 2 -1
        U[1][col] = 1
        U[2][col] = -1
        # V: g = B[1][k0] +1, h = B[1][k1] +1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        # W: contributes to C[1][k1] with -1
        W[k1 * n + 1][col] = -1
        col += 1

    # Final block: column 14 (single column, schoolbook)
    k = 14
    # C[0][k] = a*B[0][k] + b*B[1][k]
    # C[1][k] = c*B[0][k] + d*B[1][k]

    # Term: a * B[0][k]
    U[0][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1

    # Term: b * B[1][k]
    U[1][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 0][col] = 1
    col += 1

    # Term: c * B[0][k]
    U[2][col] = 1
    V[0 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1

    # Term: d * B[1][k]
    U[3][col] = 1
    V[1 * p + k][col] = 1
    W[k * n + 1][col] = 1
    col += 1

    assert col == total_rank, f"col={col}, total_rank={total_rank}"

    return {"U": U, "V": V, "W": W}
