import time
import numpy as np
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def _verify_full(U, V, W):
        """Verify full tensor decomposition"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        # Check all i, j, k, i_prime, k_prime
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i_prime in range(n):
                        for k_prime in range(p):
                            # Sum over r
                            val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k_prime*n + i_prime])
                            target = 1.0 if (i == i_prime and k == k_prime) else 0.0
                            if abs(val - target) > 1e-9:
                                return False
        return True

    def _residuals(x, R):
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        U = x[:num_u*R].reshape(num_u, R)
        V = x[num_u*R : num_u*R + num_v*R].reshape(num_v, R)
        W = x[num_u*R + num_v*R : num_u*R + num_v*R + num_w*R].reshape(num_w, R)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i_prime in range(n):
                        for k_prime in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k_prime * n + i_prime
                            val = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                            target = 1.0 if (i == i_prime and k == k_prime) else 0.0
                            res.append(val - target)
        return np.array(res)

    def _snap_to_half_int(arr):
        return np.round(2 * arr) / 2.0

    def _solve_rank(R, max_iters=100, n_restarts=20):
        num_u = n * m
        num_v = m * p
        num_w = p * n
        total_vars = (num_u + num_v + num_w) * R
        
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                return None

            # Random initialization
            U_init = np.random.uniform(-1, 1, size=(num_u, R))
            V_init = np.random.uniform(-1, 1, size=(num_v, R))
            W_init = np.random.uniform(-1, 1, size=(num_w, R))
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(_residuals, x0, args=(R,), method='trf', max_nfev=max_iters*20)
                
                if res.cost < 1e-6:
                    x_opt = res.x
                    U_opt = x_opt[:num_u*R].reshape(num_u, R)
                    V_opt = x_opt[num_u*R : num_u*R + num_v*R].reshape(num_v, R)
                    W_opt = x_opt[num_u*R + num_v*R : num_u*R + num_v*R + num_w*R].reshape(num_w, R)
                    
                    # Snap to half-integers
                    U_snap = _snap_to_half_int(U_opt)
                    V_snap = _snap_to_half_int(V_opt)
                    W_snap = _snap_to_half_int(W_opt)
                    
                    if _verify_full(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                    # Try integer snap
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if _verify_full(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                        
            except Exception:
                pass
                
        return None

    best_rank = rank_schoolbook
    best_result = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
    
    # Try ranks from 18 up to schoolbook-1
    for r in range(18, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
            
        result = _solve_rank(r, max_iters=100, n_restarts=20)
        if result is not None:
            best_rank = r
            best_result = result
            if r == 18:
                break
        
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
