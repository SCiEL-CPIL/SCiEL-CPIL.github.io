"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strategy: Use Winograd 3x3x3 (rank 23) for the 3x3x3 block,
    # and Strassen 2x2x2 (rank 7) for 2x2x2 blocks in the remaining dimensions.
    # Partition: n=3 (one block), m=5 = 3+2, p=5 = 3+2
    # Blocks: (3,3,3) Winograd 23, (3,3,2) 17, (3,2,3) 17, (3,2,2) 11
    # Total: 23 + 17 + 17 + 11 = 68

    # Strassen 2x2x2 factorization
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    # Winograd 3x3x3 factorization (rank 23)
    # This is the standard Winograd 3x3x3 scheme
    # A is 3x3, B is 3x3, C = A*B is 3x3
    # U: 9x23 (rows: a00,a01,a02,a10,a11,a12,a20,a21,a22)
    # V: 9x23 (rows: b00,b01,b02,b10,b11,b12,b20,b21,b22)
    # W: 9x23 (rows: c00,c01,c02,c10,c11,c12,c20,c21,c22)

    # The 23 rank-1 terms of the Winograd 3x3x3 scheme
    # Using the standard Winograd factorization:
    # Preprocessing A:
    #   d0 = a00+a01+a02, d1 = a00-a01, d2 = a02-a01
    #   d3 = a10+a11+a12, d4 = a10-a11, d5 = a12-a11
    #   d6 = a20+a21+a22, d7 = a20-a21, d8 = a22-a21
    # Preprocessing B:
    #   e0 = b00+b10+b20, e1 = b01+b11+b21, e2 = b02+b12+b22
    #   e3 = b00-b01, e4 = b02-b01, e5 = b10-b11, e6 = b12-b11, e7 = b20-b21, e8 = b22-b21

    # Multiplications (23 total):
    # z0 = d0*e0, z1 = d1*e1, z2 = d2*e1, z3 = d3*e0, z4 = d4*e1, z5 = d5*e1
    # z6 = d6*e0, z7 = d7*e1, z8 = d8*e1, z9 = d0*e3, z10 = d1*e3, z11 = d2*e3
    # z12 = d3*e3, z13 = d4*e3, z14 = d5*e3, z15 = d6*e3, z16 = d7*e3, z17 = d8*e3
    # z18 = d0*e5, z19 = d1*e5, z20 = d2*e5, z21 = d3*e5, z22 = d4*e5

    # Let me construct the U, V, W matrices for Winograd 3x3x3
    # Each column c corresponds to one multiplication z_c = d_i * e_j
    # U[row][c] = coefficient of A entry in d_i
    # V[row][c] = coefficient of B entry in e_j
    # W[row][c] = coefficient of z_c in C entry

    # Define the 9 d-variables (rows of U):
    # d0 = a00+a01+a02 -> U[0][c]=1, U[1][c]=1, U[2][c]=1 if c uses d0
    # d1 = a00-a01 -> U[0][c]=1, U[1][c]=-1 if c uses d1
    # d2 = a02-a01 -> U[2][c]=1, U[1][c]=-1 if c uses d2
    # d3 = a10+a11+a12 -> U[3][c]=1, U[4][c]=1, U[5][c]=1 if c uses d3
    # d4 = a10-a11 -> U[3][c]=1, U[4][c]=-1 if c uses d4
    # d5 = a12-a11 -> U[5][c]=1, U[4][c]=-1 if c uses d5
    # d6 = a20+a21+a22 -> U[6][c]=1, U[7][c]=1, U[8][c]=1 if c uses d6
    # d7 = a20-a21 -> U[6][c]=1, U[7][c]=-1 if c uses d7
    # d8 = a22-a21 -> U[8][c]=1, U[7][c]=-1 if c uses d8

    # Define the 9 e-variables (rows of V):
    # e0 = b00+b10+b20 -> V[0][c]=1, V[3][c]=1, V[6][c]=1 if c uses e0
    # e1 = b01+b11+b21 -> V[1][c]=1, V[4][c]=1, V[7][c]=1 if c uses e1
    # e2 = b02+b12+b22 -> V[2][c]=1, V[5][c]=1, V[8][c]=1 if c uses e2
    # e3 = b00-b01 -> V[0][c]=1, V[1][c]=-1 if c uses e3
    # e4 = b02-b01 -> V[2][c]=1, V[1][c]=-1 if c uses e4
    # e5 = b10-b11 -> V[3][c]=1, V[4][c]=-1 if c uses e5
    # e6 = b12-b11 -> V[5][c]=1, V[4][c]=-1 if c uses e6
    # e7 = b20-b21 -> V[6][c]=1, V[7][c]=-1 if c uses e7
    # e8 = b22-b21 -> V[8][c]=1, V[7][c]=-1 if c uses e8

    # The 23 multiplications:
    # z0 = d0*e0, z1 = d1*e1, z2 = d2*e1, z3 = d3*e0, z4 = d4*e1, z5 = d5*e1
    # z6 = d6*e0, z7 = d7*e1, z8 = d8*e1, z9 = d0*e3, z10 = d1*e3, z11 = d2*e3
    # z12 = d3*e3, z13 = d4*e3, z14 = d5*e3, z15 = d6*e3, z16 = d7*e3, z17 = d8*e3
    # z18 = d0*e5, z19 = d1*e5, z20 = d2*e5, z21 = d3*e5, z22 = d4*e5

    # W matrix: how each z_c contributes to C entries
    # C[i][j] = sum_k A[i][k] * B[k][j]
    # We need to express C[i][j] as a linear combination of the z_c's

    # This is getting complex. Let me just use a verified rank-23 factorization.
    # For now, I'll use schoolbook for the 3x3x3 block (rank 27) and see if the total is better.
    # Total: 27 + 17 + 17 + 11 = 72. Worse than 71.

    # Let me go back to the Strassen-based approach which gives 71.
    # Partition: n=[1,2], m=[2,1,2], p=[2,1,2]
    # 4 Strassen 2x2x2 blocks: rank 4*7 = 28
    # Remaining: 43 schoolbook
    # Total: 28 + 43 = 71

    def build_decomposition():
        rank_total = 71
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]

        col = 0
        # 4 Strassen 2x2x2 blocks
        blocks = [
            ([0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [2, 3]),
            ([0, 1], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3]),
        ]

        covered = set()
        for i_idx, j_idx, k_idx in blocks:
            for i in i_idx:
                for j in j_idx:
                    for k in k_idx:
                        covered.add((i, j, k))

            i0, i1 = i_idx
            j0, j1 = j_idx
            k0, k1 = k_idx

            u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]

            for c in range(7):
                for s in range(4):
                    U[u_rows[s]][col + c] = U_strassen[s][c]
                    V[v_rows[s]][col + c] = V_strassen[s][c]
                    W[w_rows[s]][col + c] = W_strassen[s][c]
            col += 7

        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

        assert col == rank_total, f"Expected rank {rank_total}, got {col}"
        return U, V, W

    U, V, W = build_decomposition()

    # Verify
    def verify(U_mat, V_mat, W_mat):
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i * m + j][c] * V_mat[j * p + k][c] * W_mat[k * n + i][c] for c in range(rank))
                    if val != 1:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
