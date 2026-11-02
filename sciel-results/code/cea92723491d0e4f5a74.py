"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _validate(U, V, W, n, m, p):
    for i_A in range(n):
        for j_A in range(m):
            for j_B in range(m):
                for k_B in range(p):
                    for i_C in range(n):
                        for k_C in range(p):
                            idx_u = i_A * m + j_A
                            idx_v = j_B * p + k_B
                            idx_w = k_C * n + i_C
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            expected = 1.0 if (i_A == i_C and j_A == j_B) else 0.0
                            if abs(val - expected) > 1e-9:
                                return False
    return True

def _als_decompose(n, m, p, rank, max_iter=50, tol=1e-10, seed=42):
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                rows = []
                rhs = []
                for j_p in range(m):
                    for k in range(p):
                        for k_p in range(p):
                            for i_p in range(n):
                                idx_v = j_p * p + k
                                idx_w = k_p * n + i_p
                                rows.append(V[idx_v, :] * W[idx_w, :])
                                rhs.append(1.0 if (i_p == i and j_p == j and k == k_p) else 0.0)
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[idx_u, :] = sol
        
        # Update V
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                rows = []
                rhs = []
                for i in range(n):
                    for j_p in range(m):
                        for k_p in range(p):
                            for i_p in range(n):
                                idx_u = i * m + j_p
                                idx_w = k_p * n + i_p
                                rows.append(U[idx_u, :] * W[idx_w, :])
                                rhs.append(1.0 if (i_p == i and j_p == j and k_p == k) else 0.0)
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[idx_v, :] = sol
        
        # Update W
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                rows = []
                rhs = []
                for i_p in range(n):
                    for j in range(m):
                        for j_p in range(m):
                            for k_p in range(p):
                                idx_u = i_p * m + j_p
                                idx_v = j * p + k_p
                                rows.append(U[idx_u, :] * V[idx_v, :])
                                rhs.append(1.0 if (i == i_p and j_p == j and k == k_p) else 0.0)
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[idx_w, :] = sol
        
        # Check error
        err = 0.0
        for i_A in range(n):
            for j_A in range(m):
                for j_B in range(m):
                    for k_B in range(p):
                        for i_C in range(n):
                            for k_C in range(p):
                                idx_u = i_A * m + j_A
                                idx_v = j_B * p + k_B
                                idx_w = k_C * n + i_C
                                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                                expected = 1.0 if (i_A == i_C and j_A == j_B) else 0.0
                                err += (val - expected) ** 2
        err = np.sqrt(err)
        if err < tol:
            break
    
    return U, V, W, err

def _round_to_half(U, V, W):
    return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    best_result = None
    best_rank = n * m * p  # 72
    
    n_sub, m_sub, p_sub = 3, 3, 4
    
    for sub_rank in [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]:
        if time.time() > deadline - 15:
            break
        
        found = False
        for seed in [42, 123, 456, 789, 111, 222, 333, 444, 555, 666]:
            if time.time() > deadline - 15:
                break
            
            U_sub, V_sub, W_sub, err = _als_decompose(n_sub, m_sub, p_sub, sub_rank, max_iter=30, tol=1e-10, seed=seed)
            
            if time.time() > deadline - 15:
                break
            
            U_sub_r, V_sub_r, W_sub_r = _round_to_half(U_sub, V_sub, W_sub)
            
            if _validate(U_sub_r, V_sub_r, W_sub_r, n_sub, m_sub, p_sub):
                total_rank = 2 * sub_rank
                U_full = np.zeros((n * m, total_rank))
                V_full = np.zeros((m * p, total_rank))
                W_full = np.zeros((p * n, total_rank))
                
                U_full[:, 0:sub_rank] = U_sub_r
                V_full[:, 0:sub_rank] = V_sub_r
                W_full[:, 0:sub_rank] = W_sub_r
                
                U_full[:, sub_rank:2*sub_rank] = U_sub_r
                for j in range(m):
                    for k in range(p_sub):
                        V_full[j * p + (p_sub + k), sub_rank:2*sub_rank] = V_sub_r[j * p_sub + k, :]
                for k in range(p_sub):
                    for i in range(n):
                        W_full[(p_sub + k) * n + i, sub_rank:2*sub_rank] = W_sub_r[k * n + i, :]
                
                if _validate(U_full, V_full, W_full, n, m, p):
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best_result = (U_full, V_full, W_full)
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
