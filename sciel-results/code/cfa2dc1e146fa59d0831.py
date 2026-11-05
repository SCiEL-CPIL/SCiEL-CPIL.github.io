import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 130.0  # Safety margin before 150s kill

    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    def discrete_local_search(U, V, W, n, m, p, max_iters=2000, time_limit=None):
        """Local search on half-integer grid to find exact solution."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        def compute_error():
            T = np.einsum('ar,br,cr->abc', U, V, W)
            # Build target
            target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = 1.0
            return np.max(np.abs(T - target))
        
        current_err = compute_error()
        if current_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist(), True
        
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            
            # Pick a random matrix
            mat_idx = rng.integers(0, 3)
            if mat_idx == 0:
                mat = U
                rows, cols = U.shape
            elif mat_idx == 1:
                mat = V
                rows, cols = V.shape
            else:
                mat = W
                rows, cols = W.shape
            
            row = rng.integers(0, rows)
            col = rng.integers(0, cols)
            old_val = mat[row, col]
            
            # Try small perturbations
            deltas = [-1.0, -0.5, 0.5, 1.0]
            best_delta = None
            best_new_err = current_err
            
            for d in deltas:
                new_val = old_val + d
                mat[row, col] = new_val
                new_err = compute_error()
                if new_err < best_new_err:
                    best_new_err = new_err
                    best_delta = d
                mat[row, col] = old_val
            
            if best_delta is not None:
                mat[row, col] = old_val + best_delta
                current_err = best_new_err
                if current_err < 1e-9:
                    break
            else:
                # No improvement, try larger steps occasionally
                if it % 10 == 0:
                    d = rng.choice([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
                    mat[row, col] = old_val + d
                    new_err = compute_error()
                    if new_err < current_err * 1.1:  # Allow some increase
                        current_err = new_err
                    else:
                        mat[row, col] = old_val
        
        final_err = compute_error()
        success = final_err < 1e-9
        return U.tolist(), V.tolist(), W.tolist(), success

    best_U, best_V, best_W, best_rank = None, None, None, 45

    from scipy.optimize import least_squares

    def residuals(params, R):
        U_flat = params[:9 * R].reshape(9, R)
        V_flat = params[9 * R:9 * R + 15 * R].reshape(15, R)
        W_flat = params[9 * R + 15 * R:].reshape(15, R)
        
        res = np.zeros(45)
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    u = U_flat[i * 3 + j]
                    v = V_flat[j * 5 + k]
                    w = W_flat[k * 3 + i]
                    res[idx] = np.dot(u, v * w) - 1.0
                    idx += 1
        return res

    def try_round_and_verify(U_sol, V_sol, W_sol, R, deadline):
        """Try various rounding strategies to find exact solution."""
        # Try rounding to various rational scales
        for scale in [1, 2, 4, 5, 8, 10, 16, 20]:
            U_r = np.round(U_sol * scale) / scale
            V_r = np.round(V_sol * scale) / scale
            W_r = np.round(W_sol * scale) / scale
            
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                return U_list, V_list, W_list, R, True
        
        # If direct rounding didn't work, try discrete local search
        if time.time() < deadline - 5:
            U_ls, V_ls, W_ls, ls_success = discrete_local_search(
                U_sol, V_sol, W_sol, 3, 3, 5,
                max_iters=1000,
                time_limit=deadline - 5
            )
            if ls_success:
                return U_ls, V_ls, W_ls, R, True
        
        return None, None, None, R, False

    # Try ranks from 40 upward (we know 45 works, try to find lower)
    target_ranks = [40, 41, 42, 43, 44]
    
    for R in target_ranks:
        if time.time() > deadline - 30:
            break
            
        found = False
        
        # Try multiple restarts with different initialization strategies
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.default_rng(1000 + R * 100 + restart * 17)
            
            # Strategy 1: Random normal initialization
            if restart < 2:
                U_init = rng.normal(size=(9, R)) * 0.3
                V_init = rng.normal(size=(15, R)) * 0.3
                W_init = rng.normal(size=(15, R)) * 0.3
            # Strategy 2: Sparse initialization (fewer non-zeros)
            elif restart == 2:
                U_init = np.zeros((9, R))
                V_init = np.zeros((15, R))
                W_init = np.zeros((15, R))
                for c in range(R):
                    # Assign each column to a random (i,j,k) pattern
                    i = rng.integers(0, 3)
                    j = rng.integers(0, 3)
                    k = rng.integers(0, 5)
                    U_init[i*3+j, c] = rng.choice([-1, 1])
                    V_init[j*5+k, c] = rng.choice([-1, 1])
                    W_init[k*3+i, c] = rng.choice([-1, 1])
            # Strategy 3: Block-structured initialization
            else:
                U_init = rng.normal(size=(9, R)) * 0.2
                V_init = rng.normal(size=(15, R)) * 0.2
                W_init = rng.normal(size=(15, R)) * 0.2
                # Add some structure
                for c in range(0, R, 3):
                    i = c % 3
                    j = (c // 3) % 3
                    k = (c // 9) % 5
                    U_init[i*3+j, c] = 1.0
                    V_init[j*5+k, c] = 1.0
                    W_init[k*3+i, c] = 1.0
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals(p, R),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=3000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14
                )
                
                if res.cost < 1e-10:
                    U_sol = res.x[:9 * R].reshape(9, R)
                    V_sol = res.x[9 * R:9 * R + 15 * R].reshape(15, R)
                    W_sol = res.x[9 * R + 15 * R:].reshape(15, R)
                    
                    U_list, V_list, W_list, R_found, success = try_round_and_verify(
                        U_sol, V_sol, W_sol, R, deadline
                    )
                    
                    if success:
                        best_U, best_V, best_W, best_rank = U_list, V_list, W_list, R_found
                        found = True
                        break
                        
            except Exception:
                pass
        
        if found:
            break

    if best_U is None:
        best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
