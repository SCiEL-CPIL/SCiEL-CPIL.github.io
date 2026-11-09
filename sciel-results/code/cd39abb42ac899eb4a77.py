"""
Rank-44 decomposition for 3x4 * 4x4 matrix multiplication.

Uses a block decomposition:
- Top 2 rows of A (2x4) multiplied by B (4x4) using Strassen-like reduction on 2x2 blocks.
  This reduces the rank from 32 to 28.
- Bottom 1 row of A (1x4) multiplied by B (4x4) using schoolbook.
  This has rank 16.
Total rank: 28 + 16 = 44.
"""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    def add_strassen(idx_u, idx_v, idx_w):
        """
        Adds a Strassen decomposition for a 2x2 matrix multiplication.
        idx_u: [u00, u01, u10, u11] indices in U
        idx_v: [v00, v01, v10, v11] indices in V
        idx_w: [w00, w01, w10, w11] indices in W
        """
        nonlocal rank
        # P1 = (x00+x11)(y00+y11)
        U[idx_u[0]][rank] += 1
        U[idx_u[3]][rank] += 1
        V[idx_v[0]][rank] += 1
        V[idx_v[3]][rank] += 1
        W[idx_w[0]][rank] += 1
        W[idx_w[3]][rank] += 1
        rank += 1

        # P2 = (x10+x11)y00
        U[idx_u[2]][rank] += 1
        U[idx_u[3]][rank] += 1
        V[idx_v[0]][rank] += 1
        W[idx_w[1]][rank] += 1
        W[idx_w[3]][rank] -= 1
        rank += 1

        # P3 = x00(y01-y11)
        U[idx_u[0]][rank] += 1
        V[idx_v[1]][rank] += 1
        V[idx_v[3]][rank] -= 1
        W[idx_w[0]][rank] += 1
        W[idx_w[2]][rank] += 1
        rank += 1

        # P4 = x11(y10-y00)
        U[idx_u[3]][rank] += 1
        V[idx_v[2]][rank] += 1
        V[idx_v[0]][rank] -= 1
        W[idx_w[0]][rank] += 1
        W[idx_w[1]][rank] += 1
        rank += 1

        # P5 = (x00+x01)y11
        U[idx_u[0]][rank] += 1
        U[idx_u[1]][rank] += 1
        V[idx_v[3]][rank] += 1
        W[idx_w[0]][rank] -= 1
        W[idx_w[2]][rank] += 1
        rank += 1

        # P6 = (x10-x00)(y00+y01)
        U[idx_u[2]][rank] += 1
        U[idx_u[0]][rank] -= 1
        V[idx_v[0]][rank] += 1
        V[idx_v[1]][rank] += 1
        W[idx_w[3]][rank] += 1
        rank += 1

        # P7 = (x01-x11)(y10+y11)
        U[idx_u[1]][rank] += 1
        U[idx_u[3]][rank] -= 1
        V[idx_v[2]][rank] += 1
        V[idx_v[3]][rank] += 1
        W[idx_w[0]][rank] += 1
        rank += 1

    def add_scalar(u_idx, v_idx, w_idx, coeff=1):
        nonlocal rank
        U[u_idx][rank] += coeff
        V[v_idx][rank] += coeff
        W[w_idx][rank] += coeff
        rank += 1

    # Block 1: C11 (top-left 2x2)
    # A1 (rows 0-1, cols 0-1) * B1 (rows 0-1, cols 0-1)
    idx_u_11 = [0, 1, 4, 5]
    idx_v_11 = [0, 1, 4, 5]
    idx_w_11 = [0, 3, 1, 4] # C[0,0]=0, C[0,1]=3, C[1,0]=1, C[1,1]=4
    add_strassen(idx_u_11, idx_v_11, idx_w_11)

    # A2 (rows 0-1, cols 2-3) * B2 (rows 2-3, cols 0-1)
    idx_u_22 = [2, 3, 6, 7]
    idx_v_22 = [8, 9, 12, 13]
    idx_w_22 = [0, 3, 1, 4]
    add_strassen(idx_u_22, idx_v_22, idx_w_22)

    # Block 2: C12 (top-right 2x2)
    # A1 (rows 0-1, cols 0-1) * B3 (rows 0-1, cols 2-3)
    idx_u_13 = [0, 1, 4, 5]
    idx_v_13 = [2, 3, 6, 7]
    idx_w_13 = [6, 9, 7, 10] # C[0,2]=6, C[0,3]=9, C[1,2]=7, C[1,3]=10
    add_strassen(idx_u_13, idx_v_13, idx_w_13)

    # A2 (rows 0-1, cols 2-3) * B4 (rows 2-3, cols 2-3)
    idx_u_24 = [2, 3, 6, 7]
    idx_v_24 = [10, 11, 14, 15]
    idx_w_24 = [6, 9, 7, 10]
    add_strassen(idx_u_24, idx_v_24, idx_w_24)

    # Block 3: C21 (bottom-left 1x2)
    # A_bottom (row 2, cols 0-1) * B1 (rows 0-1, cols 0-1)
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
    add_scalar(8, 0, 2)
    add_scalar(9, 4, 2)
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
    add_scalar(8, 1, 5)
    add_scalar(9, 5, 5)

    # A_bottom (row 2, cols 2-3) * B2 (rows 2-3, cols 0-1)
    # C[2,0] += A[2,2]B[2,0] + A[2,3]B[3,0]
    add_scalar(10, 8, 2)
    add_scalar(11, 12, 2)
    # C[2,1] += A[2,2]B[2,1] + A[2,3]B[3,1]
    add_scalar(10, 9, 5)
    add_scalar(11, 13, 5)

    # Block 4: C22 (bottom-right 1x2)
    # A_bottom (row 2, cols 0-1) * B3 (rows 0-1, cols 2-3)
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2]
    add_scalar(8, 2, 8)
    add_scalar(9, 6, 8)
    # C[2,3] = A[2,0]B[0,3] + A[2,1]B[1,3]
    add_scalar(8, 3, 11)
    add_scalar(9, 7, 11)

    # A_bottom (row 2, cols 2-3) * B4 (rows 2-3, cols 2-3)
    # C[2,2] += A[2,2]B[2,2] + A[2,3]B[3,2]
    add_scalar(10, 10, 8)
    add_scalar(11, 14, 8)
    # C[2,3] += A[2,2]B[2,3] + A[2,3]B[3,3]
    add_scalar(10, 11, 11)
    add_scalar(11, 15, 11)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
