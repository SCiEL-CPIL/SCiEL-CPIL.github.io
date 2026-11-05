import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    deadline = time.time() + 118.0
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        terms = []
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_rank7(a_row_off, b_col_off, j_off, c_row_off, c_col_off, col_start):
        for r in range(7):
            u_d, v_d, w_d = rank7_terms[r]
            c = col_start + r
            for (ai, aj), val in u_d.items():
                U[(a_row_off + ai) * m + (j_off + aj)][c] += val
            for (bj, bk), val in v_d.items():
                V[(j_off + bj) * p + (b_col_off + bk)][c] += val
            for (ci, ck), val in w_d.items():
                W[(c_col_off + ck) * n + (c_row_off + ci)][c] += val
        return col_start + 7

    def add_rank4(a_row, b_col_off, j_off, c_col_off, col_start):
        for r in range(4):
            c = col_start + r
            j_idx = j_off + (r % 2)
            k_idx = b_col_off + (r // 2)
            U[a_row * m + j_idx][c] = 1
            V[j_idx * p + k_idx][c] = 1
            W[k_idx * n + a_row][c] = 1
        return col_start + 4

    col = 0
    col = add_rank7(0, 0, 0, 0, 0, col)
    col = add_rank7(0, 0, 2, 0, 0, col)
    col = add_rank7(0, 2, 0, 0, 2, col)
    col = add_rank7(0, 2, 2, 0, 2, col)
    col = add_rank4(2, 0, 0, 0, col)
    col = add_rank4(2, 0, 2, 0, col)
    col = add_rank4(2, 2, 0, 2, col)
    col = add_rank4(2, 2, 2, 2, col)

    assert col == total_rank

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(total_rank):
        a_r = np.dot(U_arr[:, r], A_test.flatten())
        b_r = np.dot(V_arr[:, r], B_test.flatten())
        C_computed += W_arr[:, r].reshape(p, n).T * (a_r * b_r)

    if np.max(np.abs(C_computed - C_expected)) > 1e-9:
        return {"U": U_school, "V": V_school, "W": W_school}

    return {"U": U, "V": V, "W": W}
