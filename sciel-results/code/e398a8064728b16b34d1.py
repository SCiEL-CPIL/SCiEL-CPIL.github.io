"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 118.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        T_decomp = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            u = np.array(U[:, r], dtype=float)
            v = np.array(V[:, r], dtype=float)
            w = np.array(W[:, r], dtype=float)
            T_decomp += np.einsum('i,j,k->ijk', u, v, w)
        T_target = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i * m + j, j * p + k, k * n + i] = 1
        return np.allclose(T_decomp, T_target, atol=1e-9)

    def strassen_2x2x2():
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    best_result = schoolbook()
    best_rank = n * m * p

    def try_candidate(U_rows, V_rows, W_rows):
        nonlocal best_result, best_rank
        rank = len(U_rows)
        if rank >= best_rank:
            return
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            for idx, val in enumerate(U_rows[c]):
                if val:
                    U[idx][c] = val
            for idx, val in enumerate(V_rows[c]):
                if val:
                    V[idx][c] = val
            for idx, val in enumerate(W_rows[c]):
                if val:
                    W[idx][c] = val
        if verify(U, V, W, n, m, p):
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}

    def build_block_based(row_groups, mid_groups, col_groups):
        """Build decomposition: 2x2x2 blocks use Strassen, others schoolbook."""
        U_rows, V_rows, W_rows = [], [], []

        def add(u, v, w):
            U_rows.append(u)
            V_rows.append(v)
            W_rows.append(w)

        Su, Sv, Sw = strassen_2x2x2()

        for rg in row_groups:
            for mg in mid_groups:
                for cg in col_groups:
                    if len(rg) == 2 and len(mg) == 2 and len(cg) == 2:
                        i0, i1 = rg
                        j0, j1 = mg
                        k0, k1 = cg
                        for c in range(7):
                            u = [0] * (n * m)
                            v = [0] * (m * p)
                            w = [0] * (p * n)
                            u[i0 * m + j0] = Su[0][c]
                            u[i0 * m + j1] = Su[1][c]
                            u[i1 * m + j0] = Su[2][c]
                            u[i1 * m + j1] = Su[3][c]
                            v[j0 * p + k0] = Sv[0][c]
                            v[j0 * p + k1] = Sv[1][c]
                            v[j1 * p + k0] = Sv[2][c]
                            v[j1 * p + k1] = Sv[3][c]
                            w[k0 * n + i0] = Sw[0][c]
                            w[k0 * n + i1] = Sw[1][c]
                            w[k1 * n + i0] = Sw[2][c]
                            w[k1 * n + i1] = Sw[3][c]
                            add(u, v, w)
                    else:
                        for i in rg:
                            for j in mg:
                                for k in cg:
                                    u = [0] * (n * m); u[i * m + j] = 1
                                    v = [0] * (m * p); v[j * p + k] = 1
                                    w = [0] * (p * n); w[k * n + i] = 1
                                    add(u, v, w)
        return U_rows, V_rows, W_rows

    # Candidate 1: rows {0,1},{2}; mids {0,1},{2,3}; cols {0,1},{2,3},{4}
    # 2x2x2 Strassen blocks: rows{0,1} x mids{0,1} x cols{0,1} -> 7
    #                         rows{0,1} x mids{0,1} x cols{2,3} -> 7
    #                         rows{0,1} x mids{2,3} x cols{0,1} -> 7
    #                         rows{0,1} x mids{2,3} x cols{2,3} -> 7
    #                         rows{2}   x ... -> 1x2x2 schoolbook (4 each)
    row_groups = [[0, 1], [2]]
    mid_groups = [[0, 1], [2, 3]]
    col_groups = [[0, 1], [2, 3], [4]]
    try_candidate(*build_block_based(row_groups, mid_groups, col_groups))

    # Candidate 2: also group cols {3,4} to get more 2x2x2 blocks
    row_groups2 = [[0, 1], [2]]
    mid_groups2 = [[0, 1], [2, 3]]
    col_groups2 = [[0, 1], [2, 3, 4]]  # 2x2x3 -> not all 2, schoolbook for 3 cols
    try_candidate(*build_block_based(row_groups2, mid_groups2, col_groups2))

    # Candidate 3: cols {0,1},{2,3},{4} but mids singletons to test
    try_candidate(*build_block_based([[0, 1], [2]], [[0], [1], [2], [3]], [[0, 1], [2, 3], [4]]))

    if time.time() > deadline - 5.0:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
