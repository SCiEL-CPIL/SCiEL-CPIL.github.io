import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 130.0  # 20s safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
        return bool(np.allclose(T, 1.0, atol=1e-8))

    # ---- Build the rank-64 structured decomposition ----
    # 8 Strassen blocks (i in {0,1}, 2 j_blocks, 2 k_blocks, 2 k_sub) = 56
    # 4 rank-2 blocks for i=2 (j_blocks x k_blocks) = 8
    target_rank = 64
    U_new = [[0] * target_rank for _ in range(n * m)]
    V_new = [[0] * target_rank for _ in range(m * p)]
    W_new = [[0] * target_rank for _ in range(p * n)]

    col = 0

    def add_strassen_2x2x2(offset_i, offset_j, offset_k, col_start):
        col = col_start
        # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+0)][col] += 1
        U_new[(offset_i+1)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M2: (A10+A11)B00 -> C10:+1, C11:-1
        U_new[(offset_i+1)*m+(offset_j+0)][col] += 1
        U_new[(offset_i+1)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+0)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+1)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += -1
        col += 1
        # M3: A00(B10-B11) -> C01:+1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += -1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M4: A11(B01-B00) -> C00:+1, C10:+1
        U_new[(offset_i+1)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+1)][col] += 1
        V_new[(offset_j+0)*p+(offset_k+0)][col] += -1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += 1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M5: (A00+A01)B11 -> C00:-1, C01:+1
        U_new[(offset_i+0)*m+(offset_j+0)][col] += 1
        U_new[(offset_i+0)*m+(offset_j+1)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += -1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        # M7: (A01-A10)(B10+B11) -> C00:-1, C11:+1
        U_new[(offset_i+0)*m+(offset_j+1)][col] += 1
        U_new[(offset_i+1)*m+(offset_j+0)][col] += -1
        V_new[(offset_j+1)*p+(offset_k+0)][col] += 1
        V_new[(offset_j+1)*p+(offset_k+1)][col] += 1
        W_new[(offset_k+0)*n+(offset_i+0)][col] += -1
        W_new[(offset_k+1)*n+(offset_i+1)][col] += 1
        col += 1
        return col

    # 8 Strassen blocks (i in {0,1}, 2 j_blocks, 2 k_blocks, 2 k_sub)
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            for k_sub in [0, 1]:
                offset_i = 0
                offset_j = j_block * 2
                offset_k = k_block * 4 + k_sub * 2
                col = add_strassen_2x2x2(offset_i, offset_j, offset_k, col)

    # 4 rank-2 blocks for i=2 (j_blocks x k_blocks)
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            offset_i = 2
            offset_j = j_block * 2
            offset_k = k_block * 4
            # term a: j = offset_j
            U_new[offset_i*m + offset_j][col] = 1
            for kk in range(4):
                V_new[offset_j*p + (offset_k+kk)][col] = 1
                W_new[(offset_k+kk)*n + offset_i][col] = 1
            col += 1
            # term b: j = offset_j+1
            U_new[offset_i*m + (offset_j+1)][col] = 1
            for kk in range(4):
                V_new[(offset_j+1)*p + (offset_k+kk)][col] = 1
                W_new[(offset_k+kk)*n + offset_i][col] = 1
            col += 1

    best_result = {"U": U_new, "V": V_new, "W": W_new}
    best_rank = target_rank

    if not verify(U_new, V_new, W_new, target_rank):
        # Fallback to naive
        naive_rank = n * m * p
        U_naive = [[0] * naive_rank for _ in range(n * m)]
        V_naive = [[0] * naive_rank for _ in range(m * p)]
        W_naive = [[0] * naive_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j][col] = 1
                    V_naive[j * p + k][col] = 1
                    W_naive[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
        best_rank = naive_rank

    if time.time() > DEADLINE - 20:
        return best_result

    # ---- Linear-dependency elimination to reduce rank further ----
    U_arr = np.array(best_result["U"], dtype=float)
    V_arr = np.array(best_result["V"], dtype=float)
    W_arr = np.array(best_result["W"], dtype=float)
    R = best_rank
    M = np.zeros((n*m*p, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i*m*p + j*p + k
                for r in range(R):
                    M[row, r] = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]

    def eliminate(M, U, V, W, R):
        """Greedy elimination: repeatedly find a redundant column and drop it."""
        R_cur = R
        kept = list(range(R))
        improved = True
        while improved and R_cur > 1:
            improved = False
            for idx in range(len(kept)):
                r = kept[idx]
                others = [k for k in kept if k != r]
                if len(others) == 0:
                    continue
                sub = M[:, others]
                # Check if M[:, r] is in column space of sub
                try:
                    rank_sub = np.linalg.matrix_rank(sub)
                    combined = np.hstack([sub, M[:, [r]]])
                    rank_comb = np.linalg.matrix_rank(combined)
                except Exception:
                    continue
                if rank_comb == rank_sub:
                    kept.remove(r)
                    R_cur -= 1
                    improved = True
                    break
        return kept, R_cur

    kept, R_reduced = eliminate(M, U_arr, V_arr, W_arr, R)
    if R_reduced < R:
        U_red = [[U_arr[i, r] for r in kept] for i in range(n*m)]
        V_red = [[V_arr[i, r] for r in kept] for i in range(m*p)]
        W_red = [[W_arr[i, r] for r in kept] for i in range(p*n)]
        if verify(U_red, V_red, W_red, R_reduced):
            best_result = {"U": U_red, "V": V_red, "W": W_red}
            best_rank = R_reduced

    if time.time() > DEADLINE - 5:
        return best_result

    # ---- Try to add a new term that lets us drop more (local search) ----
    rng = np.random.RandomState(42)
    current = best_result
    cur_rank = best_rank
    for trial in range(300):
        if time.time() > DEADLINE - 5:
            break
        U_arr = np.array(current["U"], dtype=float)
        V_arr = np.array(current["V"], dtype=float)
        W_arr = np.array(current["W"], dtype=float)
        R_cur = cur_rank
        new_U = U_arr[:, :R_cur].copy()
        new_V = V_arr[:, :R_cur].copy()
        new_W = W_arr[:, :R_cur].copy()
        extra_U = rng.randn(n*m)
        extra_V = rng.randn(m*p)
        extra_W = rng.randn(p*n)
        new_U = np.hstack([new_U, extra_U.reshape(-1,1)])
        new_V = np.hstack([new_V, extra_V.reshape(-1,1)])
        new_W = np.hstack([new_W, extra_W.reshape(-1,1)])
        R_try = R_cur + 1
        M_try = np.zeros((n*m*p, R_try))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i*m*p + j*p + k
                    for r in range(R_try):
                        M_try[row, r] = new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
        kept2, R2 = eliminate(M_try, new_U, new_V, new_W, R_try)
        if R2 < cur_rank:
            U2 = [[new_U[i, r] for r in kept2] for i in range(n*m)]
            V2 = [[new_V[i, r] for r in kept2] for i in range(m*p)]
            W2 = [[new_W[i, r] for r in kept2] for i in range(p*n)]
            if verify(U2, V2, W2, R2):
                current = {"U": U2, "V": V2, "W": W2}
                cur_rank = R2
                best_result = current
                best_rank = R2

    return best_result
