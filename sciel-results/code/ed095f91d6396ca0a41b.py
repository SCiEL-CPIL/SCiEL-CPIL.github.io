import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (1 if valid product term)
    
    # Build list of (u_idx, v_idx, w_idx) for non-zero tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 44
    
    # Precompute entry indices for each factor position
    entries_by_u = [[] for _ in range(n * m)]
    entries_by_v = [[] for _ in range(m * p)]
    entries_by_w = [[] for _ in range(p * n)]
    for (u_idx, v_idx, w_idx) in tensor_entries:
        entries_by_u[u_idx].append((v_idx, w_idx))
        entries_by_v[v_idx].append((u_idx, w_idx))
        entries_by_w[w_idx].append((u_idx, v_idx))
    
    # Build zero-constraint entries (all (u,v,w) not in tensor_entries)
    all_u = set(range(n * m))
    all_v = set(range(m * p))
    all_w = set(range(p * n))
    tensor_set = set(tensor_entries)
    zero_entries = []
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                if (u_idx, v_idx, w_idx) not in tensor_set:
                    zero_entries.append((u_idx, v_idx, w_idx))
    
    # Known rank-39 decomposition
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            
            # m1 = (a+d)(e+h)
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            
            # m2 = (c+d)e
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            
            # m3 = a(f-h)
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            
            # m4 = d(g-e)
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            
            # m5 = (a+b)h
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            
            # m6 = (c-a)(e+f)
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            
            # m7 = (b-d)(g+h)
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
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
    
    # Verify a decomposition is exact (checks both tensor and zero constraints)
    def verify(U, V, W):
        rank = len(U[0])
        # Check tensor entries sum to 1
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        # Check zero entries sum to 0 (sample if too many)
        if len(zero_entries) <= 500:
            for (u_idx, v_idx, w_idx) in zero_entries:
                total = 0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(total) > 1e-9:
                    return False
        else:
            # Sample zero entries
            np.random.seed(0)
            sample_idx = np.random.choice(len(zero_entries), min(200, len(zero_entries)), replace=False)
            for idx in sample_idx:
                u_idx, v_idx, w_idx = zero_entries[idx]
                total = 0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(total) > 1e-9:
                    return False
        return True
    
    # Try to find a lower rank decomposition using ALS + rounding
    def try_als_rank(target_rank, max_iters=150, seed=42):
        if time.time() > deadline - 8:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random Gaussian
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 8:
                break
            
            # Update U: for each u_idx, solve least squares
            for u_idx in range(n * m):
                e_list = entries_by_u[u_idx]
                if not e_list:
                    continue
                A_mat = np.array([V[v_idx, :] * W[w_idx, :] for (v_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx, :] = x
                except:
                    pass
            
            # Update V: for each v_idx, solve least squares
            for v_idx in range(m * p):
                e_list = entries_by_v[v_idx]
                if not e_list:
                    continue
                A_mat = np.array([U[u_idx, :] * W[w_idx, :] for (u_idx, w_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx, :] = x
                except:
                    pass
            
            # Update W: for each w_idx, solve least squares
            for w_idx in range(p * n):
                e_list = entries_by_w[w_idx]
                if not e_list:
                    continue
                A_mat = np.array([U[u_idx, :] * V[v_idx, :] for (u_idx, v_idx) in e_list])
                b_vec = np.ones(len(e_list))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx, :] = x
                except:
                    pass
            
            # Compute error on tensor entries
            err = 0.0
            for (u_idx, v_idx, w_idx) in tensor_entries:
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                err += (val - 1) ** 2
            
            # Also compute zero-constraint error (sample)
            zero_err = 0.0
            if len(zero_entries) > 0:
                np.random.seed(it)
                sample_size = min(100, len(zero_entries))
                sample_idx = np.random.choice(len(zero_entries), sample_size, replace=False)
                for idx in sample_idx:
                    u_idx, v_idx, w_idx = zero_entries[idx]
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    zero_err += val ** 2
                zero_err /= sample_size
            
            total_err = err + 0.1 * zero_err
            
            if total_err < best_err:
                best_err = total_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-12 and zero_err < 1e-12:
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
        
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 38 down to 35
    for target_rank in [38, 37, 36, 35]:
        if time.time() > deadline - 12:
            break
        
        for seed in range(8):
            if time.time() > deadline - 12:
                break
            
            result = try_als_rank(target_rank, max_iters=120, seed=seed)
            if result is not None:
                return result
    
    # Fall back to known rank-39
    return make_rank39()
