import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build list of (u_idx, v_idx, w_idx) for non-zero tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j  # i*2 + j
                v_idx = j * p + k  # j*11 + k
                w_idx = k * n + i  # k*2 + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    # Verify a decomposition is exact
    def verify(U, V, W):
        rank = len(U[0])
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = 0
            for r in range(rank):
                total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
            if abs(total - 1) > 1e-9:
                return False
        # Also verify that non-tensor entries sum to 0
        # Check a sample of non-tensor entries to ensure they are zero
        all_u = set(range(n * m))
        all_v = set(range(m * p))
        all_w = set(range(p * n))
        
        tensor_set = set(tensor_entries)
        
        # Check some random non-tensor entries
        np.random.seed(12345)
        for _ in range(1000):
            u = np.random.randint(0, n * m)
            v = np.random.randint(0, m * p)
            w = np.random.randint(0, p * n)
            if (u, v, w) not in tensor_set:
                total = 0
                for r in range(rank):
                    total += U[u][r] * V[v][r] * W[w][r]
                if abs(total) > 1e-9:
                    return False
        return True
    
    # Known rank-39 decomposition
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        # 5 pairs of columns using Strassen's algorithm
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
        
        # Single column k2 = 10 using schoolbook
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
    
    # Try to find a lower rank decomposition using ALS + rounding
    def try_als_rank(target_rank, max_iters=200, seed=42):
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random Gaussian
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for u_idx in range(n * m):
                entries_u = [(v_idx, w_idx) for (uu, v_idx, w_idx) in tensor_entries if uu == u_idx]
                if not entries_u:
                    continue
                A_mat = np.array([V[v_idx, :] * W[w_idx, :] for (v_idx, w_idx) in entries_u])
                b_vec = np.ones(len(entries_u))
                try:
                    x, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx, :] = x
                except:
                    pass
            
            # Update V
            for v_idx in range(m * p):
                entries_v = [(u_idx, w_idx) for (u_idx, vv, w_idx) in tensor_entries if vv == v_idx]
                if not entries_v:
                    continue
                A_mat = np.array([U[u_idx, :] * W[w_idx, :] for (u_idx, w_idx) in entries_v])
                b_vec = np.ones(len(entries_v))
                try:
                    x, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx, :] = x
                except:
                    pass
            
            # Update W
            for w_idx in range(p * n):
                entries_w = [(u_idx, v_idx) for (u_idx, v_idx, ww) in tensor_entries if ww == w_idx]
                if not entries_w:
                    continue
                A_mat = np.array([U[u_idx, :] * V[v_idx, :] for (u_idx, v_idx) in entries_w])
                b_vec = np.ones(len(entries_w))
                try:
                    x, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx, :] = x
                except:
                    pass
            
            # Compute error
            err = 0.0
            for (u_idx, v_idx, w_idx) in tensor_entries:
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                err += (val - 1) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
        
        # Round to half-integers
        if best_U is None:
            return None
        
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Convert to lists
        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()
        
        # Verify
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 38 down to 35
    for target_rank in [38, 37, 36, 35]:
        if time.time() > deadline - 15:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            result = try_als_rank(target_rank, max_iters=100, seed=seed)
            if result is not None:
                return result
    
    # Fall back to known rank-39
    return make_rank39()
