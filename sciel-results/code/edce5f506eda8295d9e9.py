import numpy as np
import time
import itertools
import scipy.optimize as so

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m   # 6
    V_size = m * p   # 9
    W_size = p * n   # 6

    # Build target tensor (only 18 non-zero entries)
    T = np.zeros((U_size, V_size, W_size))
    eqs = []  # (u_idx, v_idx, w_idx) where T should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                eqs.append((u_idx, v_idx, w_idx))

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)

    def verify_int(U_int, V_int, W_int):
        """Verify exact decomposition with integer entries (representing half-integers * 2).
        U_int, V_int, W_int are integer arrays.
        Condition: for each (u,v,w) in eqs: sum_c U[u,c]*V[v,c]*W[w,c] == 8
        For other (u,v,w): sum == 0
        """
        rank = U_int.shape[1]
        # Check all 6*9*6 = 324 combinations
        for u in range(U_size):
            for v in range(V_size):
                for w in range(W_size):
                    s = 0
                    for c in range(rank):
                        s += U_int[u, c] * V_int[v, c] * W_int[w, c]
                    target = 8 if (u, v, w) in eqs else 0
                    if s != target:
                        return False
        return True

    def verify_float(U, V, W):
        """Verify exact decomposition with half-integer entries (float arrays)."""
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9):
            return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9):
            return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9):
            return False
        rank = U.shape[1]
        for u, v, w in eqs:
            s = np.sum(U[u, :] * V[v, :] * W[w, :])
            if abs(s - 1.0) > 1e-9:
                return False
        # Check zeros
        for u in range(U_size):
            for v in range(V_size):
                for w in range(W_size):
                    if (u, v, w) not in eqs:
                        s = np.sum(U[u, :] * V[v, :] * W[w, :])
                        if abs(s) > 1e-9:
                            return False
        return True

    def snap_and_verify(U, V, W):
        """Snap to nearest half-integer and verify"""
        U_int = np.round(U * 2).astype(int)
        V_int = np.round(V * 2).astype(int)
        W_int = np.round(W * 2).astype(int)
        if verify_int(U_int, V_int, W_int):
            return U_int / 2.0, V_int / 2.0, W_int / 2.0
        return None

    def exhaustive_search(R, time_limit):
        """Direct exhaustive search over {-1,0,1} patterns for small R."""
        if time.time() > deadline - 5:
            return None
        search_start = time.time()
        
        # For R=11,12, search over {-1,0,1}^R for each column
        # Use pruning: check partial constraints early
        # Strategy: enumerate U patterns, then for each, enumerate V patterns,
        # and solve for W or check consistency
        
        # Simpler: full enumeration with early termination
        # For R=11: 3^11 = 177147 per matrix, too many for triple product
        # Instead: enumerate U columns, V columns, and for each pair, check if W can satisfy
        # Actually for small R, let's do structured search
        
        # Approach: enumerate all U in {-1,0,1}^(6*R) with pruning
        # For each U, enumerate V in {-1,0,1}^(9*R), and for each W equation,
        # the constraint is linear in W entries
        # This is complex. Let's do a smarter approach:
        
        # For R=11: try all combinations of U[:,0], V[:,0], W[:,0] etc.
        # Actually, let's just do random sampling with verification for speed
        
        # Better approach for small R: use the fact that we need sum_c U[u,c]*V[v,c]*W[w,c] = 8
        # for 18 specific (u,v,w) and 0 for 306 others
        # Try random {-1,0,1} patterns with verification
        
        attempts = 0
        while time.time() < search_start + time_limit and time.time() < deadline - 10:
            attempts += 1
            U_int = np.random.choice([-1, 0, 1], size=(U_size, R))
            V_int = np.random.choice([-1, 0, 1], size=(V_size, R))
            W_int = np.random.choice([-1, 0, 1], size=(W_size, R))
            
            if verify_int(U_int, V_int, W_int):
                return U_int / 2.0, V_int / 2.0, W_int / 2.0
        
        return None

    def als_decompose(rank, max_iter=300, tol=1e-14):
        """ALS with symmetry breaking: fix U[0,0]=1"""
        U = np.random.randn(U_size, rank) * 0.7
        V = np.random.randn(V_size, rank) * 0.7
        W = np.random.randn(W_size, rank) * 0.7
        # Symmetry breaking: scale so U[0,0] = 1
        if abs(U[0, 0]) > 1e-6:
            scale = 1.0 / U[0, 0]
            U *= scale
        else:
            U[0, 0] = 1.0
        
        prev_cost = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_cost = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Update U (fix U[0,0]=1)
            for u in range(U_size):
                M = (V[:, None, :] * W[None, :, :]).reshape(-1, rank)
                b = T[u, :, :].reshape(-1)
                if u == 0:
                    # Solve with constraint U[0,0]=1
                    # U[0,:] = M[0,:]^{-1} * b[0] but with U[0,0]=1
                    # Actually, just solve normally and rescale
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    if abs(sol[0]) > 1e-6:
                        sol = sol / sol[0]
                    else:
                        sol[0] = 1.0
                    U[0, :] = sol
                else:
                    sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                    U[u, :] = sol
            
            # Update V
            for v in range(V_size):
                M = (U[:, None, :] * W[None, :, :]).reshape(-1, rank)
                b = T[:, v, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                V[v, :] = sol
            
            # Update W
            for w in range(W_size):
                M = (U[:, None, :] * V[None, :, :]).reshape(-1, rank)
                b = T[:, :, w].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                W[w, :] = sol
            
            # Calculate cost
            cost = 0.0
            for u, v, w in eqs:
                s = np.sum(U[u, :] * V[v, :] * W[w, :])
                cost += (s - 1.0) ** 2
            for u in range(U_size):
                for v in range(V_size):
                    for w in range(W_size):
                        if (u, v, w) not in eqs:
                            s = np.sum(U[u, :] * V[v, :] * W[w, :])
                            cost += s ** 2
            
            if cost < best_cost:
                best_cost = cost
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
        
        U, V, W = best_U, best_V, best_W
        
        result = snap_and_verify(U, V, W)
        if result is not None:
            return result
        
        # Polish with scipy if close
        if best_cost < 1e-3:
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def residuals_func(x):
                U_f = x[:U_size*rank].reshape(U_size, rank)
                V_f = x[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                W_f = x[U_size*rank+V_size*rank:].reshape(W_size, rank)
                res = []
                for u, v, w in eqs:
                    res.append(np.sum(U_f[u] * V_f[v] * W_f[w]) - 1.0)
                for u in range(U_size):
                    for v in range(V_size):
                        for w in range(W_size):
                            if (u, v, w) not in eqs:
                                res.append(np.sum(U_f[u] * V_f[v] * W_f[w]))
                return np.array(res)
            
            try:
                res_polish = so.least_squares(residuals_func, x0, method='lm', max_nfev=3000, ftol=1e-18, xtol=1e-18, gtol=1e-18)
                x_opt = res_polish.x
                U_opt = x_opt[:U_size*rank].reshape(U_size, rank)
                V_opt = x_opt[U_size*rank:U_size*rank+V_size*rank].reshape(V_size, rank)
                W_opt = x_opt[U_size*rank+V_size*rank:].reshape(W_size, rank)
                
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result is not None:
                    return result
            except Exception:
                pass
        
        return None

    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_size, rank))
        V = np.zeros((V_size, rank))
        W = np.zeros((W_size, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try ranks from 11 upwards (theoretical minimum for 2x3x3)
    for R in [11, 12, 13, 14, 15, 16, 17]:
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time()
        if time_left < 10:
            break
        
        # Exhaustive/random integer search first (fast, reliable for small R)
        exhaust_time = min(15.0, time_left * 0.3)
        result = exhaustive_search(R, exhaust_time)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # ALS search
        time_left = deadline - time.time()
        if time_left < 10:
            break
        max_restarts = min(50, int(time_left / 2.0))
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            result = als_decompose(R)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
