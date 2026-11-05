import numpy as np
import time
from scipy.optimize import least_squares, differential_evolution

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 130.0

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
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

    def try_rational_rounding(U_sol, V_sol, W_sol, R):
        for scale in [1, 2, 4, 8, 10, 20]:
            U_r = np.round(U_sol * scale) / scale
            V_r = np.round(V_sol * scale) / scale
            W_r = np.round(W_sol * scale) / scale
            
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                return U_list, V_list, W_list, True
        
        # Try half-integers specifically
        for scale in [2, 4]:
            U_r = np.round(U_sol * scale) / scale
            V_r = np.round(V_sol * scale) / scale
            W_r = np.round(W_sol * scale) / scale
            
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            if verify_decomposition(U_list, V_list, W_list, 3, 3, 5):
                return U_list, V_list, W_list, True
        
        return None, None, None, False

    best_U, best_V, best_W, best_rank = None, None, None, 45

    # Strategy: Try to find rank-44 decomposition using DFT-inspired structure
    # The 3x3x5 matrix multiplication can be viewed as 5 slices of 3x3
    # Apply DFT over Z_5 to each of the 3x3 slices
    
    # For 3x3x3, Strassen's algorithm gives rank 7
    # For 3x3x5, if we use DFT, we get 5 frequency components
    # Each frequency component is a 3x3 multiplication
    # If we use rank 7 for each, total rank would be 35
    # But this doesn't work directly because the DFT mixes things
    
    # Let's try a more direct approach: use known rank-44 structure if it exists
    # Or use numerical optimization to find a rank-44 solution
    
    # First, try rank 44 with multiple restarts
    target_ranks = [44, 43, 42, 41, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 30:
            break
            
        found = False
        
        # Multiple restarts with different random seeds
        for restart in range(5):
            if time.time() > deadline - 15:
                break
                
            rng = np.random.default_rng(1234 + R * 7 + restart * 11)
            U_init = rng.normal(size=(9, R)) * 0.5
            V_init = rng.normal(size=(15, R)) * 0.5
            W_init = rng.normal(size=(15, R)) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals(p, R),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=10000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14
                )
                
                if res.cost < 1e-10:
                    U_sol = res.x[:9 * R].reshape(9, R)
                    V_sol = res.x[9 * R:9 * R + 15 * R].reshape(15, R)
                    W_sol = res.x[9 * R + 15 * R:].reshape(15, R)
                    
                    U_r, V_r, W_r, success = try_rational_rounding(U_sol, V_sol, W_sol, R)
                    
                    if success:
                        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, R
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    # If rank 44-40 not found, try using differential evolution for rank 44
    if best_rank >= 45 and time.time() < deadline - 20:
        R = 44
        # Use differential evolution for global optimization
        bounds = [(-5, 5)] * (9 * R + 15 * R + 15 * R)
        
        def objective(params):
            return np.sum(residuals(params, R) ** 2)
        
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                tol=1e-8,
                seed=42,
                polish=False
            )
            
            if de_result.fun < 1e-6:
                U_sol = de_result.x[:9 * R].reshape(9, R)
                V_sol = de_result.x[9 * R:9 * R + 15 * R].reshape(15, R)
                W_sol = de_result.x[9 * R + 15 * R:].reshape(15, R)
                
                # Polish with least_squares
                res = least_squares(
                    lambda p: residuals(p, R),
                    de_result.x,
                    method='lm',
                    x_scale='jac',
                    max_nfev=5000,
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14
                )
                
                if res.cost < 1e-10:
                    U_sol = res.x[:9 * R].reshape(9, R)
                    V_sol = res.x[9 * R:9 * R + 15 * R].reshape(15, R)
                    W_sol = res.x[9 * R + 15 * R:].reshape(15, R)
                    
                    U_r, V_r, W_r, success = try_rational_rounding(U_sol, V_sol, W_sol, R)
                    
                    if success:
                        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, R
        except Exception:
            pass
    
    # If still not found, try rank 45 (schoolbook)
    if best_U is None:
        best_U, best_V, best_W, best_rank = make_schoolbook(3, 3, 5)
    
    return {"U": best_U, "V": best_V, "W": best_W}
