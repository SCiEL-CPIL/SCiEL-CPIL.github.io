import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # We will build a rank-54 decomposition by applying Strassen's 7-term
    # algorithm to all six disjoint 2x2x2 sub-blocks in the j-k plane.
    # j pairs: (0,1), (2,3); k pairs: (0,1), (2,3), (4,5)
    # Each block saves 1 rank compared to the 8 standard terms.
    # Total rank = 60 - 6 = 54.

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Identify all 2x2x2 blocks to apply Strassen to
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    # Set of (i, j, k) tuples covered by Strassen blocks (to skip in standard terms)
    strassen_cells = set()
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            for i in range(n):
                for j in (j1, j2):
                    for k in (k1, k2):
                        strassen_cells.add((i, j, k))

    col = 0

    # Add standard (rank-8) terms for cells NOT in any Strassen block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_cells:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # Add Strassen terms for each 2x2x2 block
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # Index computations for this block
            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2

            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2

            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1

            # Strassen's 7 products for C = A*B where
            # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
            # P1 = (a00+a11)(b00+b11)
            # P2 = (a10+a11)b00
            # P3 = a00(b01-b11)
            # P4 = a11(b10-b00)
            # P5 = (a00+a01)b11
            # P6 = (a10-a00)(b00+b01)
            # P7 = (a01-a11)(b10+b11)
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6

            # P1
            U[u00][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            W[w11][col] += 1
            col += 1

            # P2
            U[u10][col] += 1
            U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1
            W[w11][col] += -1
            col += 1

            # P3
            U[u00][col] += 1
            V[v01][col] += 1
            V[v11][col] += -1
            W[w01][col] += 1
            W[w11][col] += 1
            col += 1

            # P4
            U[u11][col] += 1
            V[v10][col] += 1
            V[v00][col] += -1
            W[w00][col] += 1
            W[w10][col] += 1
            col += 1

            # P5
            U[u00][col] += 1
            U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1
            W[w01][col] += 1
            col += 1

            # P6
            U[u10][col] += 1
            U[u00][col] += -1
            V[v00][col] += 1
            V[v01][col] += 1
            W[w11][col] += 1
            col += 1

            # P7
            U[u01][col] += 1
            U[u11][col] += -1
            V[v10][col] += 1
            V[v11][col] += 1
            W[w00][col] += 1
            col += 1

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback: return a rank-60 trivial decomposition
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U, "V": V, "W": W}
