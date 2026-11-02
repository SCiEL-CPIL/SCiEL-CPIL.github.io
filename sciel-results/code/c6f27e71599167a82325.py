"""Construct a rank-21 exact decomposition for 2x2 @ 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Decompose p=9 into p1=2 and p2=7.
    # For p1=2: use Laderman decomposition (rank 7) for 2x2 @ 2x2.
    # For p2=7: use schoolbook decomposition (rank 14) for 2x2 @ 2x7.
    # Total rank = 7 + 14 = 21.

    rank1 = 7  # Laderman for 2x2 @ 2x2
    rank2 = 14  # Schoolbook for 2x2 @ 2x7
    total_rank = rank1 + rank2

    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * total_rank for _ in range(m * p)]  # 18 x 21
    W = [[0] * total_rank for _ in range(p * n)]  # 18 x 21

    # --- Part 1: Laderman decomposition for columns 0,1 of B (2x2 @ 2x2) ---
    # Laderman's 7-term decomposition for multiplying two 2x2 matrices.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # The 7 bilinear forms:
    # t1 = a11 * b11
    # t2 = a12 * b22
    # t3 = (a11 + a12) * (b21 + b22)
    # t4 = (a11 + a22) * (b11 + b12)
    # t5 = (a21 + a22) * (b11 + b12)
    # t6 = (a21 + a22) * (b21 + b22)
    # t7 = a22 * b11
    #
    # C11 = t4 - t5 + t1 - t7
    # C12 = t3 - t4 + t2 - t1
    # C21 = t6 - t5 + t7 - t2
    # C22 = t4 - t3 + t5 - t6

    # U indices: 0=a11, 1=a12, 2=a21, 3=a22
    # V indices for first 2 columns of B: 0=b11, 1=b12, 2=b21, 3=b22
    # W indices for first 2 columns of C: 0=c11, 1=c12, 2=c21, 3=c22

    # t1 = a11 * b11
    U[0][0] = 1
    V[0][0] = 1
    W[0][0] = 1  # C11 += t1

    # t2 = a12 * b22
    U[1][1] = 1
    V[3][1] = 1
    W[1][1] = 1  # C12 += t2

    # t3 = (a11 + a12) * (b21 + b22)
    U[0][2] = 1
    U[1][2] = 1
    V[2][2] = 1
    V[3][2] = 1
    W[1][2] = 1  # C12 += t3

    # t4 = (a11 + a22) * (b11 + b12)
    U[0][3] = 1
    U[3][3] = 1
    V[0][3] = 1
    V[1][3] = 1
    W[0][3] = 1  # C11 += t4

    # t5 = (a21 + a22) * (b11 + b12)
    U[2][4] = 1
    U[3][4] = 1
    V[0][4] = 1
    V[1][4] = 1
    W[2][4] = 1  # C21 += t5

    # t6 = (a21 + a22) * (b21 + b22)
    U[2][5] = 1
    U[3][5] = 1
    V[2][5] = 1
    V[3][5] = 1
    W[2][5] = 1  # C21 += t6

    # t7 = a22 * b11
    U[3][6] = 1
    V[0][6] = 1
    W[0][6] = 1  # C11 += t7

    # Now set the W coefficients correctly:
    # C11 = t4 - t5 + t1 - t7
    W[0][0] = 1   # t1
    W[0][3] = 1   # t4
    W[0][4] = -1  # t5
    W[0][6] = -1  # t7

    # C12 = t3 - t4 + t2 - t1
    W[1][0] = -1  # t1
    W[1][1] = 1   # t2
    W[1][2] = 1   # t3
    W[1][3] = -1  # t4

    # C21 = t6 - t5 + t7 - t2
    W[2][1] = -1  # t2
    W[2][4] = -1  # t5
    W[2][5] = 1   # t6
    W[2][6] = 1   # t7

    # C22 = t4 - t3 + t5 - t6
    W[3][2] = -1  # t3
    W[3][3] = 1   # t4
    W[3][4] = 1   # t5
    W[3][5] = -1  # t6

    # --- Part 2: Schoolbook for columns 2..8 of B (2x2 @ 2x7) ---
    # For each column k in 2..8 of B, we need 4 products.
    # U indices: 0=a11, 1=a12, 2=a21, 3=a22
    # V indices: 2*p+k for row 2 of B, 2*p+1+k for row 1 of B
    # W indices: k*n+0 for C[k,0], k*n+1 for C[k,1]

    col_offset = 7
    for k in range(2, 9):
        # a11 * B[0,k] -> C[0,k]
        U[0][col_offset] = 1
        V[k][col_offset] = 1
        W[k * n + 0][col_offset] = 1
        col_offset += 1

        # a12 * B[1,k] -> C[0,k]
        U[1][col_offset] = 1
        V[p + k][col_offset] = 1
        W[k * n + 0][col_offset] = 1
        col_offset += 1

        # a21 * B[0,k] -> C[1,k]
        U[2][col_offset] = 1
        V[k][col_offset] = 1
        W[k * n + 1][col_offset] = 1
        col_offset += 1

        # a22 * B[1,k] -> C[1,k]
        U[3][col_offset] = 1
        V[p + k][col_offset] = 1
        W[k * n + 1][col_offset] = 1
        col_offset += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
