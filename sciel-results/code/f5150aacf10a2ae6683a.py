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
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    R = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(R):
                    s += Ua[i * m + j, t] * Va[j * p + k, t] * Wa[k * n + i, t]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def _round_half(a):
    return np.round(2 * a) / 2.0

def _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline):
    """
    Attempts to reduce the rank by finding a lower-rank exact decomposition.
    Uses multiple strategies:
    1. Direct linear algebra on the tensor equations
    2. Optimization with multiple restarts
    3. Column sampling from the known solution
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R0 = Ua.shape[1]
    
    if R0 <= target_rank:
        return U, V, W

    # Strategy 1: Try to solve the linear system directly
    # The tensor equation is: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # This is nonlinear, but we can try to find a solution using the known structure
    
    # Strategy 2: Optimization with multiple restarts
    from scipy.optimize import minimize
    
    def loss(params):
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        T = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
        err = T - 1.0
        return np.sum(err**2)

    def gradient(params):
        U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        T = np.einsum('it,jt,kt->ijk', U_flat, V_flat, W_flat)
        err = T - 1.0
        
        dU = 2.0 * np.einsum('ijk,jt,kt->it', err, V_flat, W_flat)
        dV = 2.0 * np.einsum('ijk,it,kt->jt', err, U_flat, W_flat)
        dW = 2.0 * np.einsum('ijk,it,jt->kt', err, U_flat, V_flat)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    # Try multiple initializations
    best_x = None
    best_fun = np.inf
    
    # Initialization 1: Use first target_rank columns from known solution
    U_keep = Ua[:, :target_rank].copy()
    V_keep = Va[:, :target_rank].copy()
    W_keep = Wa[:, :target_rank].copy()
    init_params = np.concatenate([U_keep.flatten(), V_keep.flatten(), W_keep.flatten()])
    
    if time.time() < deadline - 20:
        res = minimize(loss, init_params, jac=gradient, method='L-BFGS-B', 
                     options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    
    # Initialization 2: Random perturbations of the kept columns
    rng = np.random.default_rng(42)
    for attempt in range(3):
        if time.time() > deadline - 15:
            break
        
        U_rand = Ua[:, :target_rank].copy() + rng.normal(0, 0.1, size=(n*m, target_rank))
        V_rand = Va[:, :target_rank].copy() + rng.normal(0, 0.1, size=(m*p, target_rank))
        W_rand = Wa[:, :target_rank].copy() + rng.normal(0, 0.1, size=(p*n, target_rank))
        init_params = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
        
        res = minimize(loss, init_params, jac=gradient, method='L-BFGS-B', 
                     options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    
    # Initialization 3: Sample random columns from the known solution
    if time.time() < deadline - 15:
        col_indices = rng.choice(R0, size=target_rank, replace=False)
        U_sample = Ua[:, col_indices].copy()
        V_sample = Va[:, col_indices].copy()
        W_sample = Wa[:, col_indices].copy()
        init_params = np.concatenate([U_sample.flatten(), V_sample.flatten(), W_sample.flatten()])
        
        res = minimize(loss, init_params, jac=gradient, method='L-BFGS-B', 
                     options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x
    
    # If we found a good solution, try to polish it
    if best_x is not None and best_fun < 1e-4 and time.time() < deadline - 10:
        res2 = minimize(loss, best_x, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40})
        if res2.fun < best_fun:
            best_fun = res2.fun
            best_x = res2.x
    
    # Final polish if we got very close
    if best_x is not None and best_fun < 1e-6 and time.time() < deadline - 5:
        res3 = minimize(loss, best_x, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30})
        if res3.fun < best_fun:
            best_fun = res3.fun
            best_x = res3.x
    
    if best_x is not None and best_fun < 1e-6:
        new_U = best_x[:n*m*target_rank].reshape(n*m, target_rank)
        new_V = best_x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        new_W = best_x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        # Try rounding to integers first, then half-integers
        for round_func in [np.round, _round_half]:
            new_U_r = round_func(new_U)
            new_V_r = round_func(new_V)
            new_W_r = round_func(new_W)
            
            if _validate(new_U_r.tolist(), new_V_r.tolist(), new_W_r.tolist(), n, m, p):
                return new_U_r.tolist(), new_V_r.tolist(), new_W_r.tolist()
            
            # Also try mixed rounding: some integers, some half-integers
            if round_func is np.round:
                # Try rounding each matrix independently
                for u_round in [np.round, _round_half]:
                    for v_round in [np.round, _round_half]:
                        for w_round in [np.round, _round_half]:
                            new_U_mr = u_round(new_U)
                            new_V_mr = v_round(new_V)
                            new_W_mr = w_round(new_W)
                            if _validate(new_U_mr.tolist(), new_V_mr.tolist(), new_W_mr.tolist(), n, m, p):
                                return new_U_mr.tolist(), new_V_mr.tolist(), new_W_mr.tolist()
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s margin before 150s limit
    
    # Start from known 66
    known = _build_known_66()
    U, V, W = known["U"], known["V"], known["W"]
    
    if not _validate(U, V, W, n, m, p):
        return known

    # Try to reduce to 64
    if time.time() < deadline - 30:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 64, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
                
    # Try to reduce to 62
    if time.time() < deadline - 30:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 62, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try to reduce to 60
    if time.time() < deadline - 30:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 60, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try to reduce to 58
    if time.time() < deadline - 30:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 58, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Try to reduce to 56
    if time.time() < deadline - 30:
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, 56, deadline)
        if U_new is not U:
            U, V, W = U_new, V_new, W_new
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
