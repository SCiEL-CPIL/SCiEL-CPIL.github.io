import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _build_known_66():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0

    def add_rank1(u_idx, u_vals, v_idx, v_vals, w_idx, w_vals):
        nonlocal current_rank
        if current_rank >= rank:
            return
        for idx, val in zip(u_idx, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_idx, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_idx, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])

    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])

    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)

    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    import numpy as np
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    # T[i*m+j, j*p+k, k*n+i]
    # We want sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1 if (i,j,k) is a valid entry, else 0?
    # No, the tensor T is 1 at (i,j,k) for all i,j,k.
    # The matrix multiplication tensor is T[i,j,k] = 1.
    # So we check if the sum is 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Attempts to reduce the rank by selecting a subset of columns and re-solving.
    Uses multiple restarts and a penalty for non-half-integer entries.
    """
    import numpy as np
    from scipy.optimize import minimize
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    if R0 <= target_rank:
        return U, V, W
    
    def snap_half(a):
        return np.round(2 * a) / 2.0
    
    def make_solution(params):
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        return U_flat, V_flat, W_flat
    
    def half_int_penalty(a):
        nearest = np.round(2 * a) / 2.0
        return np.sum((a - nearest)**2)
    
    def loss(params, penalty_weight=0.0):
        U_flat, V_flat, W_flat = make_solution(params)
        T = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
        err = T - 1.0
        data_loss = np.sum(err**2)
        if penalty_weight > 0:
            penalty = penalty_weight * (half_int_penalty(U_flat) + half_int_penalty(V_flat) + half_int_penalty(W_flat))
            return data_loss + penalty
        return data_loss
    
    best_solution = None
    best_loss = float('inf')
    
    # Generate different column subsets
    subsets = []
    # First: keep first target_rank columns
    subsets.append(list(range(target_rank)))
    # Second: keep last target_rank columns
    subsets.append(list(range(R0 - target_rank, R0)))
    # Random subsets
    np.random.seed(42)
    for _ in range(6):
        cols = np.random.choice(R0, target_rank, replace=False).tolist()
        subsets.append(cols)
    
    for cols in subsets:
        if time.time() > deadline - 20:
            break
        
        U_keep = Ua[:, cols].copy()
        V_keep = Va[:, cols].copy()
        W_keep = Wa[:, cols].copy()
        
        init_params = np.concatenate([U_keep.flatten(), V_keep.flatten(), W_keep.flatten()])
        
        # Try different penalty weights
        for pw in [0.0, 0.01, 0.1]:
            if time.time() > deadline - 20:
                break
            
            def loss_pw(params, _pw=pw):
                return loss(params, _pw)
            
            try:
                res = minimize(loss_pw, init_params, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                continue
            
            if res.fun < best_loss:
                best_loss = res.fun
                U_opt, V_opt, W_opt = make_solution(res.x)
                
                # Snap to half-integers
                U_snapped = snap_half(U_opt)
                V_snapped = snap_half(V_opt)
                W_snapped = snap_half(W_opt)
                
                # Validate snapped version
                if _validate(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                    best_solution = (U_snapped, V_snapped, W_snapped)
                    break
                elif _validate(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p):
                    best_solution = (U_opt, V_opt, W_opt)
                    break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        return best_solution[0].tolist(), best_solution[1].tolist(), best_solution[2].tolist()
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    # Start from known 66
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known

    # Try to reduce to various ranks
    for target_rank in [64, 63, 62, 60]:
        if time.time() < deadline - 30:
            U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline)
            if U_new is not U:
                U, V, W = U_new, V_new, W_new
                if _validate(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
