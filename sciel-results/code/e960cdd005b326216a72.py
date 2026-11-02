import numpy as np
import time
import random

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size

    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M2
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M4
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1

                # M5
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1

                # M6
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M7
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1

    def is_valid_decomposition(U_arr, V_arr, W_arr, n, m, p):
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    total = 0
                    for t in range(U_arr.shape[1]):
                        total += U_arr[i * m + k, t] * V_arr[k * p + j, t] * W_arr[j * n + i, t]
                    if total != 1:
                        return False
        return True

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try multiple random shuffles and greedy removals
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        indices = list(range(rank))
        random.shuffle(indices)
        
        U_shuf = U[:, indices]
        V_shuf = V[:, indices]
        W_shuf = W[:, indices]
        
        keep_cols = list(range(U_shuf.shape[1]))
        
        for t in range(U_shuf.shape[1] - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if t not in keep_cols:
                continue
            
            new_keep = [c for c in keep_cols if c != t]
            if len(new_keep) == 0:
                break
            
            U_test = U_shuf[:, new_keep]
            V_test = V_shuf[:, new_keep]
            W_test = W_shuf[:, new_keep]
            
            if is_valid_decomposition(U_test, V_test, W_test, n, m, p):
                keep_cols = new_keep
        
        if len(keep_cols) < best_rank:
            best_rank = len(keep_cols)
            best_U = U_shuf[:, keep_cols]
            best_V = V_shuf[:, keep_cols]
            best_W = W_shuf[:, keep_cols]

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
