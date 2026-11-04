"""Matrix multiplication 2x4x8 via Strassen with optimized rank reduction."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 blocks
    n_blocks_i = 1
    n_blocks_j = 2
    n_blocks_k = 4
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for bi in range(n_blocks_i):
        for bj in range(n_blocks_j):
            for bk in range(n_blocks_k):
                i0 = 2 * bi
                i1 = 2 * bi + 1
                j0 = 2 * bj
                j1 = 2 * bj + 1
                k0 = 2 * bk
                k1 = 2 * bk + 1
                
                u_a = i0 * m + j0
                u_b = i0 * m + j1
                u_c = i1 * m + j0
                u_d = i1 * m + j1
                
                v_e = j0 * p + k0
                v_f = j0 * p + k1
                v_g = j1 * p + k0
                v_h = j1 * p + k1
                
                w_c00 = k0 * n + i0
                w_c01 = k1 * n + i0
                w_c10 = k0 * n + i1
                w_c11 = k1 * n + i1
                
                # M1 = (a+d)(e+h)
                U[u_a][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M2 = (c+d)e
                U[u_c][col] += 1; U[u_d][col] += 1
                V[v_e][col] += 1
                W[w_c10][col] += 1; W[w_c11][col] -= 1
                col += 1
                
                # M3 = a(f-h)
                U[u_a][col] += 1
                V[v_f][col] += 1; V[v_h][col] -= 1
                W[w_c01][col] += 1; W[w_c11][col] += 1
                col += 1
                
                # M4 = d(g-e)
                U[u_d][col] += 1
                V[v_g][col] += 1; V[v_e][col] -= 1
                W[w_c00][col] += 1; W[w_c10][col] += 1
                col += 1
                
                # M5 = (a+b)h
                U[u_a][col] += 1; U[u_b][col] += 1
                V[v_h][col] += 1
                W[w_c00][col] -= 1; W[w_c01][col] += 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[u_c][col] += 1; U[u_a][col] -= 1
                V[v_e][col] += 1; V[v_f][col] += 1
                W[w_c11][col] += 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[u_b][col] += 1; U[u_d][col] -= 1
                V[v_g][col] += 1; V[v_h][col] += 1
                W[w_c00][col] += 1
                col += 1
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build true tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1.0
    
    def compute_tensor(U_m, V_m, W_m, cols):
        if len(cols) == 0:
            return np.zeros((n * m, m * p, p * n))
        # Use einsum for efficiency
        Uc = U_m[:, cols]
        Vc = V_m[:, cols]
        Wc = W_m[:, cols]
        return np.einsum('ic,jc,kc->ijk', Uc, Vc, Wc)
    
    # Verify initial
    T_cur = compute_tensor(U_np, V_np, W_np, list(range(rank)))
    if not np.allclose(T_cur, T_true, atol=1e-10):
        return {"U": U, "V": V, "W": W}
    
    # Greedy elimination
    keep = list(range(rank))
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        for c in sorted(keep):
            if time.time() > deadline - 15:
                break
            new_keep = [k for k in keep if k != c]
            T_test = compute_tensor(U_np, V_np, W_np, new_keep)
            if np.allclose(T_test, T_true, atol=1e-10):
                keep = new_keep
                improved = True
    
    # Try pairs
    if time.time() < deadline - 20 and len(keep) > 32:
        keep_set = set(keep)
        pairs = list(combinations(sorted(keep_set), 2))
        for pair in pairs:
            if time.time() > deadline - 15:
                break
            new_keep = [k for k in keep if k not in pair]
            T_test = compute_tensor(U_np, V_np, W_np, new_keep)
            if np.allclose(T_test, T_true, atol=1e-10):
                keep = new_keep
                keep_set = set(keep)
                pairs = list(combinations(sorted(keep_set), 2))
    
    # Final
    if len(keep) < rank:
        U_reduced = U_np[:, keep].tolist()
        V_reduced = V_np[:, keep].tolist()
        W_reduced = W_np[:, keep].tolist()
        T_test = compute_tensor(np.array(U_reduced), np.array(V_reduced), np.array(W_reduced), list(range(len(keep))))
        if np.allclose(T_test, T_true, atol=1e-10):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    return {"U": U, "V": V, "W": W}
