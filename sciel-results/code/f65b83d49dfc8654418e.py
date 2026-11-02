import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def elapsed():
        return time.time() - start_time

    def check_decomposition(Um, Vm, Wm):
        R = Um.shape[1]
        if R == 0:
            return False
        Um_3d = Um.reshape(n, m, R)
        Vm_3d = Vm.reshape(m, p, R)
        Wm_3d = Wm.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', Um_3d, Vm_3d, Wm_3d, optimize=True)
        return np.allclose(T, 1.0, atol=1e-7)

    # Fallback: Schoolbook 216
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1

    # Solution 1: Strassen 2x2 x Schoolbook 3x3 (Rank 189)
    def build_strassen_schoolbook():
        # Strassen 2x2 factors (7 terms)
        # U_s: 4x7, V_s: 4x7, W_s: 4x7
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # M0 = (A11+A22)(B12-B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[1, 0] = 1; V_s[3, 0] = -1
        W_s[0, 0] = 1; W_s[3, 0] = 1
        # M1 = (A21+A22)B11
        U_s[2, 1] = 1; U_s[3, 1] = 1
        V_s[0, 1] = 1
        W_s[1, 1] = 1; W_s[3, 1] = -1
        # M2 = A11(B22-B12)
        U_s[0, 2] = 1
        V_s[2, 2] = 1; V_s[1, 2] = -1
        W_s[1, 2] = 1; W_s[3, 2] = 1
        # M3 = A22(B11-B22)
        U_s[3, 3] = 1
        V_s[0, 3] = 1; V_s[2, 3] = -1
        W_s[0, 3] = 1; W_s[3, 3] = 1
        # M4 = (A11+A12)B22
        U_s[0, 4] = 1; U_s[1, 4] = 1
        V_s[2, 4] = 1
        W_s[0, 4] = -1; W_s[2, 4] = 1
        # M5 = (A21-A11)(B11+B12)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        V_s[0, 5] = 1; V_s[1, 5] = 1
        W_s[2, 5] = 1
        # M6 = (A12-A22)(B21+B22)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        V_s[2, 6] = 1; V_s[3, 6] = 1
        W_s[3, 6] = 1

        # Schoolbook 3x3 factors (27 terms)
        U_l = np.zeros((9, 27))
        V_l = np.zeros((9, 27))
        W_l = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = i * 3 + j
                    U_l[i * 3 + j, col] = 1
                    V_l[j * 3 + k, col] = 1
                    W_l[k * 3 + i, col] = 1

        # Combine
        U_total = np.kron(U_s, U_l)
        V_total = np.kron(V_s, V_l)
        W_total = np.kron(W_s, W_l)
        
        # Permute rows
        perm_U = np.zeros(36, dtype=int)
        perm_V = np.zeros(36, dtype=int)
        perm_W = np.zeros(36, dtype=int)
        
        for i_b in range(2):
            for j_b in range(2):
                for i_l in range(3):
                    for j_l in range(3):
                        # Global index
                        i = i_b * 3 + i_l
                        j = j_b * 3 + j_l
                        idx = i * 6 + j
                        # Kron index
                        r_s = i_b * 2 + j_b
                        r_l = i_l * 3 + j_l
                        kron_idx = r_s * 9 + r_l
                        perm_U[idx] = kron_idx
                        perm_V[idx] = kron_idx
                        perm_W[idx] = kron_idx

        U_total = U_total[perm_U, :]
        V_total = V_total[perm_V, :]
        W_total = W_total[perm_W, :]
        
        return U_total, V_total, W_total

    U_bs, V_bs, W_bs = build_strassen_schoolbook()
    if check_decomposition(U_bs, V_bs, W_bs):
        best_U, best_V, best_W = U_bs, V_bs, W_bs
        best_rank = U_bs.shape[1]
    else:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0

    # Greedy Pruning
    def greedy_prune(Um, Vm, Wm, order, time_budget):
        R = Um.shape[1]
        U_cur, V_cur, W_cur = Um.copy(), Vm.copy(), Wm.copy()
        cur_rank = R
        for idx in order:
            if time.time() > time_budget:
                break
            if idx >= cur_rank:
                continue
            keep = [r for r in range(cur_rank) if r != idx]
            U_new = U_cur[:, keep]
            V_new = V_cur[:, keep]
            W_new = W_cur[:, keep]
            if check_decomposition(U_new, V_new, W_new):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                cur_rank = len(keep)
        return U_cur, V_cur, W_cur, cur_rank

    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        order = list(range(best_rank))
        random.shuffle(order)
        U_r, V_r, W_r, rank_r = greedy_prune(best_U, best_V, best_W, order, deadline - 20)
        if rank_r < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank_r

    if not check_decomposition(best_U, best_V, best_W):
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist(),
    }
