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

    # Strategy: Use structured initialization based on Kronecker products and
    # exploit the 3x3x5 structure by decomposing into 5 slices of 3x3

    # First, try to construct a rank-44 solution using a perturbation approach
    # Start from schoolbook and try to merge two terms

    def construct_rank44_candidate():
        """
        Try to construct a rank-44 decomposition by merging two schoolbook terms.
        The idea is to find U, V, W such that we can save one rank dimension.
        """
        R = 44
        # Start with schoolbook for first 43 terms
        U = np.zeros((9, R))
        V = np.zeros((15, R))
        W = np.zeros((15, R))
        
        idx = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    if idx >= 43:
                        break
                    U[i * 3 + j, idx] = 1
                    V[j * 5 + k, idx] = 1
                    W[k * 3 + i, idx] = 1
                    idx += 1
        
        # For the 44th term, we need to handle the remaining (i,j,k) tuples
        # There are 45 total, so we have 2 left to cover with 1 rank dimension
        # This is impossible with just 1 term, so we need a different approach
        
        # Instead, let's use a more sophisticated approach:
        # Use the fact that 3x3x5 can be seen as 5 copies of 3x3
        # Apply a linear transformation to reduce rank
        
        # For now, return None to indicate we need numerical optimization
        return None, None, None

    # Use numerical optimization with better initialization
    target_ranks = [44, 43, 42, 41, 40]
    
    for R in target_ranks:
        if time.time() > deadline - 30:
            break
            
        found = False
        
        # Multiple restarts with different random seeds and initialization strategies
        for restart in range(8):
            if time.time() > deadline - 15:
                break
                
            rng = np.random.default_rng(1234 + R * 7 + restart * 11)
            
            # Strategy 1: Random initialization
            if restart < 4:
                U_init = rng.normal(size=(9, R)) * 0.3
                V_init = rng.normal(size=(15, R)) * 0.3
                W_init = rng.normal(size=(15, R)) * 0.3
            else:
                # Strategy 2: Sparse initialization with some structure
                U_init = np.zeros((9, R))
                V_init = np.zeros((15, R))
                W_init = np.zeros((15, R))
                
                # Initialize with random sparse patterns
                for r in range(R):
                    # Each rank term covers a subset of the tensor entries
                    n_u = rng.integers(1, 4)
                    n_v = rng.integers(1, 6)
                    n_w = rng.integers(1, 6)
                    
                    u_indices = rng.choice(9, size=n_u, replace=False)
                    v_indices = rng.choice(15, size=n_v, replace=False)
                    w_indices = rng.choice(15, size=n_w, replace=False)
                    
                    U_init[u_indices, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
                    V_init[v_indices, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
                    W_init[w_indices, r] = rng.choice([-1, 1]) * rng.uniform(0.5, 2.0)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    lambda p: residuals(p, R),
                    x0,
                    method='lm',
                    x_scale='jac',
                    max_nfev=15000,
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15
                )
                
                if res.cost < 1e-12:
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
    
    # If rank 44-40 not found with least_squares, try differential evolution
    if best_rank >= 45 and time.time() < deadline - 25:
        R = 44
        
        def objective(params):
            return np.sum(residuals(params, R) ** 2)
        
        # Use differential evolution with better bounds
        bounds = [(-3, 3)] * (9 * R + 15 * R + 15 * R)
        
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
                tol=1e-10,
                seed=42,
                polish=True,
                popsize=20
            )
            
            if de_result.fun < 1e-8:
                U_sol = de_result.x[:9 * R].reshape(9, R)
                V_sol = de_result.x[9 * R:9 * R + 15 * R].reshape(15, R)
                W_sol = de_result.x[9 * R + 15 * R:].reshape(15, R)
                
                # Polish with least_squares
                res = least_squares(
                    lambda p: residuals(p, R),
                    de_result.x,
                    method='lm',
                    x_scale='jac',
                    max_nfev=10000,
                    ftol=1e-15,
                    xtol=1e-15,
                    gtol=1e-15
                )
                
                if res.cost < 1e-12:
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
