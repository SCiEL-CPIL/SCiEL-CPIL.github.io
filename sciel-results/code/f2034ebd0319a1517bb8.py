"""Matrix multiplication 2x3x8 decomposition."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 8

    # We'll use a block decomposition:
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x8) into B1 (2x8) and B2 (1x8)
    # C = A1*B1 + A2*B2
    #
    # For A1*B1: A1 is 2x2, B1 is 2x8. 
    # Split B1 into 4 pairs of 2x4 blocks: B1 = [B1_0, B1_1, B1_2, B1_3] where each is 2x4
    # Actually, let's think of this as 2x2 times 2x8.
    # We can view the 8 columns as 4 pairs. For each pair, we do a 2x2x2 Strassen.
    # That gives 4 * 7 = 28 rank terms.
    #
    # For A2*B2: A2 is 2x1, B2 is 1x8. This is outer product: C[i,k] = A2[i,0]*B2[0,k]
    # This is rank 2*8 = 16.
    #
    # Total rank: 28 + 16 = 44.

    # Let's build this decomposition.
    # Variables:
    # A is 2x3: indices i in {0,1}, j in {0,1,2}
    # B is 3x8: indices j in {0,1,2}, k in {0,...,7}
    # C is 2x8: indices i in {0,1}, k in {0,...,7}
    #
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:3] (2x1)
    # B1 = B[0:2, :] (2x8), B2 = B[2:3, :] (1x8)
    #
    # C = A1 * B1 + A2 * B2
    #
    # For A1 * B1: Let's denote the elements of A1 as a_ij for i in {0,1}, j in {0,1}
    # and B1 as b_jk for j in {0,1}, k in {0,...,7}
    #
    # We group columns of B1 into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair (k0, k0+1), we compute a 2x2 matrix product using Strassen (rank 7)
    #
    # Strassen for 2x2x2:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) b00
    # M3 = a00 (b10 - b11)
    # M4 = a11 (b01 - b00)
    # M5 = (a00 + a01) b11
    # M6 = (a10 - a00)(b00 + b10)
    # M7 = (a01 - a11)(b01 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # In our case, for each pair (k0, k0+1), the "b" values are from column k0 and k0+1 of B1.
    # b00 = B[0,k0], b01 = B[0,k0+1], b10 = B[1,k0], b11 = B[1,k0+1]
    #
    # The "a" values are the same for all pairs: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    #
    # So for each pair, we have 7 products. Total 28 products.
    #
    # For A2 * B2: A2 = A[:, 2] (2x1), B2 = B[2, :] (1x8)
    # C2[i,k] = A[2,i] * B[2,k] (using 0-indexed: A[i,2] * B[2,k])
    # This is 16 rank-1 terms.

    # Let's build the factor matrices.
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    # Strassen part: 28 terms
    # For pair index t in 0..3, columns k0 = 2*t, k1 = 2*t+1
    col = 0
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        # M1 = (a00 + a11)(b00 + b11)
        # U: a00 -> U[0][col]=1, a11 -> U[3][col]=1
        # V: b00 -> V[0*8+k0][col]=1, b11 -> V[1*8+k1][col]=1
        # W: C00 -> W[k0*2+0][col]=1, C01 -> W[k1*2+0][col]=1, C10 -> W[k0*2+1][col]=1, C11 -> W[k1*2+1][col]=1
        U[0][col] = 1   # a00
        U[3][col] = 1   # a11
        V[0 * 8 + k0][col] = 1   # b00
        V[1 * 8 + k1][col] = 1   # b11
        W[k0 * 2 + 0][col] = 1   # C00
        W[k1 * 2 + 0][col] = 1   # C01
        W[k0 * 2 + 1][col] = 1   # C10
        W[k1 * 2 + 1][col] = 1   # C11
        col += 1

        # M2 = (a10 + a11) b00
        U[1][col] = 1   # a10
        U[3][col] = 1   # a11
        V[0 * 8 + k0][col] = 1   # b00
        W[k0 * 2 + 1][col] = 1   # C10
        W[k1 * 2 + 1][col] = -1  # C11
        col += 1

        # M3 = a00 (b10 - b11)
        U[0][col] = 1   # a00
        V[1 * 8 + k0][col] = 1   # b10
        V[1 * 8 + k1][col] = -1  # b11
        W[k1 * 2 + 0][col] = 1   # C01
        W[k1 * 2 + 1][col] = 1   # C11
        col += 1

        # M4 = a11 (b01 - b00)
        U[3][col] = 1   # a11
        V[0 * 8 + k1][col] = 1   # b01
        V[0 * 8 + k0][col] = -1  # b00
        W[k0 * 2 + 0][col] = 1   # C00
        W[k0 * 2 + 1][col] = 1   # C10
        col += 1

        # M5 = (a00 + a01) b11
        U[0][col] = 1   # a00
        U[1][col] = 1   # a01
        V[1 * 8 + k1][col] = 1   # b11
        W[k0 * 2 + 0][col] = -1  # C00
        W[k1 * 2 + 0][col] = 1   # C01
        col += 1

        # M6 = (a10 - a00)(b00 + b10)
        U[1][col] = 1   # a10
        U[0][col] = -1  # a00
        V[0 * 8 + k0][col] = 1   # b00
        V[1 * 8 + k0][col] = 1   # b10
        W[k1 * 2 + 1][col] = 1   # C11
        col += 1

        # M7 = (a01 - a11)(b01 + b11)
        U[1][col] = 1   # a01
        U[3][col] = -1  # a11
        V[0 * 8 + k1][col] = 1   # b01
        V[1 * 8 + k1][col] = 1   # b11
        W[k0 * 2 + 0][col] = 1   # C00
        col += 1

    # Outer product part: 16 terms
    # C2[i,k] = A[i,2] * B[2,k]
    # U: A[i,2] -> U[i*3+2][col]
    # V: B[2,k] -> V[2*8+k][col]
    # W: C[i,k] -> W[k*2+i][col]
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col] = 1   # A[i,2]
            V[2 * 8 + k][col] = 1   # B[2,k]
            W[k * 2 + i][col] = 1   # C[i,k]
            col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
