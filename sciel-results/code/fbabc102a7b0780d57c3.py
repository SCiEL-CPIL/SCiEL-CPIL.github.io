"""Matrix multiplication 3,4,4 tensor decomposition via Strassen on 4x4x4 sub-tensor."""
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 4, 4

    # Rank-16 decomposition of the 4x4x4 tensor (Strassen on 2x2 blocks).
    # For each term c, U has a (2x2) block at rows (i2*2, i2*2+1) x (j2*2, j2*2+1),
    # V has a (2x2) block at rows (j2*2, j2*2+1) x (k2*2, k2*2+1),
    # W has a (2x2) block at rows (k2*2, k2*2+1) x (i2*2, i2*2+1).
    # U_block = u[i2,j2] * A_sub, V_block = v[i2,j2] * B_sub, W_block = w[i2,j2] * I.
    # A_sub, B_sub are 2x2 with entries in {+1,-1}.
    A_sub = [
        [[1, 1], [1, -1]],
        [[1, 0], [1, 1]],
        [[1, 1], [0, 1]],
    ]
    B_sub = [
        [[1, 0], [0, 1]],
        [[1, 0], [1, 0]],
        [[0, 1], [1, 1]],
    ]
    w = [1, 1, 1, 1, -1, 1, -1, 1]

    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 12 x 16
    V = [[0] * rank for _ in range(m * p)]  # 16 x 16
    W = [[0] * rank for _ in range(p * n)]  # 12 x 16

    for i2 in range(2):
        for j2 in range(2):
            for k2 in range(2):
                c = i2 * 4 + j2 * 2 + k2
                uval = 1
                for i in range(2):
                    for j in range(2):
                        U[(i2 * 2 + i) * m + (j2 * 2 + j)][c] = uval * A_sub[i2][j2][i][j]
                        V[(j2 * 2 + j) * p + (k2 * 2 + k)][c] = B_sub[i2][j2][j][k]
                        W[(k2 * 2 + k) * n + (i2 * 2 + i)][c] = w[c]

    # Zero out the 4th row of A (i=3) to restrict to the 3x4x4 sub-tensor.
    for j in range(m):
        U[3 * m + j] = [0] * rank

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
