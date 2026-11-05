"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strassen 2x2x2 factors
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

    # Partition: n=[1,2], m=[2,1,2], p=[2,1,2]
    # Strassen 2x2x2 blocks where all three dimensions are size 2:
    #   Block 1: i=[1,2], j=[0,1], k=[0,1]
    #   Block 2: i=[1,2], j=[0,1], k=[3,4]
    #   Block 3: i=[1,2], j=[3,4], k=[0,1]
    #   Block 4: i=[1,2], j=[3,4], k=[3,4]
    # Each saves 1 rank: 75 - 4 = 71

    strassen_blocks = [
        ([1, 2], [0, 1], [0, 1]),
        ([1, 2], [0, 1], [3, 4]),
        ([1, 2], [3, 4], [0, 1]),
        ([1, 2], [3, 4], [3, 4]),
    ]

    # Build set of (i,j,k) tuples covered by Strassen blocks
    covered = set()
    for i_list, j_list, k_list in strassen_blocks:
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    covered.add((i, j, k))

    rank_total = 4 * 7 + (n * m * p - 4 * 8)
    assert rank_total == 71, f"Expected 71, got {rank_total}"

    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0
    for i_list, j_list, k_list in strassen_blocks:
        i0, i1 = i_list
        j0, j1 = j_list
        k0, k1 = k_list

        u_rows = [i0 * m + j0, i0 * m + j1, i1 * m + j0, i1 * m + j1]
        v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
        w_rows = [k0 * n + i0, k1 * n + i0, k0 * n + i1, k1 * n + i1]

        for c in range(7):
            for s in range(4):
                U[u_rows[s]][col + c] = U_strassen[s][c]
                V[v_rows[s]][col + c] = V_strassen[s][c]
                W[w_rows[s]][col + c] = W_strassen[s][c]
        col += 7

    # Schoolbook for remaining terms
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

    # Verify correctness
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        U = np.array(U_mat)
        V = np.array(V_mat)
        W = np.array(W_mat)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    if not verify_decomposition(U, V, W, n, m, p):
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

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
