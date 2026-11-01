import time
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    # T[i, j, k] = 1 if C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # Pre-compute indices for fast ALS
    entries_by_u = [[] for _ in range(n * m)]
    entries_by_v = [[] for _ in range(m * p)]
    entries_by_w = [[] for _ in range(p * n)]
    for (u_idx, v_idx, w_idx) in tensor_entries:
        entries_by_u[u_idx].append((v_idx, w_idx))
        entries_by_v[v_idx].append((u_idx, w_idx))
        entries_by_w[w_idx].append((u_idx, v_idx))
        
    # Set of non-zero entries for quick lookup
    nz_set = set(tensor_entries)
    
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # 5 blocks of size 2 in p dimension, using 7 mults each = 35
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0      # B[0, k0]
            v_f = k1      # B[0, k1]
            v_g = 11 + k0 # B[1, k0]
            v_h = 11 + k1 # B[1, k1]
            
            w_c00 = 2 * k0     # C[0, k0]
            w_c10 = 2 * k0 + 1 # C[1, k0]
            w_c01 = 2 * k1     # C[0, k1]
            w_c11 = 2 * k1 + 1 # C[1, k1]
            
            # Strassen-like 2x2 * 2x2 block decomposition
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Last block k=10 (only 1 column)
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_full(U, V, W):
        rank = len(U[0])
        U_int = [[int(round(x * 2)) for x in row] for row in U]
        V_int = [[int(round(x * 2)) for x in row] for row in V]
        W_int = [[int(round(x * 2)) for x in row] for row in W]
        
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U_int[u_idx][r] * V_int[v_idx][r] * W_int[w_idx][r]
            if total != 8:
                return False
                
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) in nz_set:
                        continue
                    total = 0
                    for r in range(rank):
                        total += U_int[u_idx][r] * V_int[v_idx][r] * W_int[w_idx][r]
                    if total != 0:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute squared error on non-zero entries + sampled zero entries"""
        rank = len(U[0])
        err = 0.0
        for (u_idx, v_idx, w_idx) in tensor_entries:
            val = 0.0
            for r in range(rank):
                val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            err += (val - 1.0) ** 2
        # Sample 100 random zero entries
        np.random.seed(12345)
        for _ in range(100):
            u_idx = np.random.randint(n * m)
            v_idx = np.random.randint(m * p)
            w_idx = np.random.randint(p * n)
            if (u_idx, v_idx, w_idx) not in nz_set:
                val = 0.0
                for r in range(rank):
                    val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                err += val ** 2
        return err
    
    def try_als_rank(target_rank, max_iters=80, seed=42, init_type='random'):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        if init_type == 'random':
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        elif init_type == 'strassen_perturb':
            r39 = make_rank39()
            U_full = np.array(r39['U'], dtype=float)
            V_full = np.array(r39['V'], dtype=float)
            W_full = np.array(r39['W'], dtype=float)
            U = U_full[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.1
            V = V_full[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.1
            W = W_full[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.1
        else:
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # ALS Step - update U
            for u_idx in range(n * m):
                e_list = entries_by_u[u_idx]
                if not e_list: continue
                A_mat = np.array([V[v_idx] * W[w_idx] for (v_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx] = x
                except: pass
            
            # Update V
            for v_idx in range(m * p):
                e_list = entries_by_v[v_idx]
                if not e_list: continue
                A_mat = np.array([U[u_idx] * W[w_idx] for (u_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx] = x
                except: pass
            
            # Update W
            for w_idx in range(p * n):
                e_list = entries_by_w[w_idx]
                if not e_list: continue
                A_mat = np.array([U[u_idx] * V[v_idx] for (u_idx, v_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx] = x
                except: pass
            
            # Compute error
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            err = compute_error(U_list, V_list, W_list)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12:
                break
        
        if best_U is None:
            return None
        
        # Round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        if verify_full(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Discrete Local Search
        if time.time() < deadline - 15:
            dls_result = discrete_local_search(U_list, V_list, W_list, target_rank)
            if dls_result:
                return dls_result
        
        return None
    
    def discrete_local_search(U, V, W, rank):
        U_curr = [row[:] for row in U]
        V_curr = [row[:] for row in V]
        W_curr = [row[:] for row in W]
        
        if verify_full(U_curr, V_curr, W_curr):
            return {"U": U_curr, "V": V_curr, "W": W_curr}
        
        max_dls_iters = 800
        np.random.seed(99)
        
        for restart in range(8):
            if time.time() > deadline - 10:
                return None
                
            U_t = [row[:] for row in U]
            V_t = [row[:] for row in V]
            W_t = [row[:] for row in W]
            
            for step in range(max_dls_iters):
                if time.time() > deadline - 10:
                    return None
                
                factor = np.random.choice(3)
                if factor == 0:
                    idx = np.random.choice(n * m)
                    col = np.random.choice(rank)
                    old_val = U_t[idx][col]
                elif factor == 1:
                    idx = np.random.choice(m * p)
                    col = np.random.choice(rank)
                    old_val = V_t[idx][col]
                else:
                    idx = np.random.choice(p * n)
                    col = np.random.choice(rank)
                    old_val = W_t[idx][col]
                
                nearest_half = np.round(old_val * 2) / 2
                candidates = [nearest_half - 0.5, nearest_half, nearest_half + 0.5]
                candidates = [c for c in candidates if -2 <= c <= 2]
                if not candidates:
                    candidates = [0.0]
                    
                new_val = np.random.choice(candidates)
                
                if new_val == old_val:
                    continue
                    
                if factor == 0:
                    U_t[idx][col] = new_val
                elif factor == 1:
                    V_t[idx][col] = new_val
                else:
                    W_t[idx][col] = new_val
                    
                if verify_full(U_t, V_t, W_t):
                    return {"U": U_t, "V": V_t, "W": W_t}
                
                err = compute_error(U_t, V_t, W_t)
                if err < 1e-9:
                    if verify_full(U_t, V_t, W_t):
                        return {"U": U_t, "V": V_t, "W": W_t}
                
        return None
    
    # Search for lower ranks using the splitting method insight
    # Try ranks 38, 37, 36 with multiple seeds
    for target_rank in [38, 37, 36]:
        if time.time() > deadline - 20:
            break
            
        for seed in range(15):
            if time.time() > deadline - 20:
                break
                
            init_type = 'random' if seed % 3 != 0 else 'strassen_perturb'
            
            result = try_als_rank(target_rank, max_iters=60, seed=seed, init_type=init_type)
            if result is not None:
                return result
                
    # Fallback
    return make_rank39()
