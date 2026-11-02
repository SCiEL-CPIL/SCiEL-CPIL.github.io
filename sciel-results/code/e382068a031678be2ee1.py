import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    # Target tensor indices
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    u_idxs = []
    v_idxs = []
    w_idxs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idxs.append(i * m + j)
                v_idxs.append(j * p + k)
                w_idxs.append(k * n + i)
                
    u_idxs = np.array(u_idxs)
    v_idxs = np.array(v_idxs)
    w_idxs = np.array(w_idxs)
    num_constraints = len(u_idxs) # 24

    def compute_error(U, V, W):
        """Compute sum of squared errors for the tensor decomposition."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # We need sum_r U[u, r] * V[v, r] * W[w, r] == 1 for each constraint (u,v,w)
        U_expanded = U[u_idxs, :] # (24, R)
        V_expanded = V[v_idxs, :] # (24, R)
        W_expanded = W[w_idxs, :] # (24, R)
        
        products = U_expanded * V_expanded * W_expanded # (24, R)
        sums = np.sum(products, axis=1) # (24,)
        errors = sums - 1.0
        return np.sum(errors**2)

    def round_to_halfint(X):
        return np.round(X * 2) / 2

    def refine_integer(U, V, W, max_iters=2000):
        """
        Refine a half-integer solution using coordinate descent on the integer grid.
        Assumes U, V, W are close to half-integers.
        """
        U = round_to_halfint(U)
        V = round_to_halfint(V)
        W = round_to_halfint(W)
        
        current_err = compute_error(U, V, W)
        
        if current_err < 1e-9:
            return U, V, W, True
            
        # Pre-flatten for fast indexing
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        sizes = [U.size, V.size, W.size]
        total_size = sum(sizes)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try a few random updates
            for _ in range(10):
                var_idx = random.randint(0, total_size - 1)
                
                if var_idx < sizes[0]:
                    r, c = np.unravel_index(var_idx, U.shape)
                    old_val = U[r, c]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        U[r, c] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U[r, c] = old_val
                            
                elif var_idx < sizes[0] + sizes[1]:
                    v_idx = var_idx - sizes[0]
                    r, c = np.unravel_index(v_idx, V.shape)
                    old_val = V[r, c]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        V[r, c] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V[r, c] = old_val
                            
                else:
                    w_idx = var_idx - sizes[0] - sizes[1]
                    r, c = np.unravel_index(w_idx, W.shape)
                    old_val = W[r, c]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        W[r, c] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W[r, c] = old_val
                            
            if not improved:
                break
                
            if current_err < 1e-9:
                return U, V, W, True
                
        return U, V, W, False

    def try_rank_numeric(R, num_restarts=10, seed=0):
        """
        Try to find a rank-R solution using numerical optimization + integer refinement.
        """
        if time.time() > deadline - 20:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
                
            np.random.seed(seed + restart)
            
            # Initialize U, V, W with small random values
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U = x[:U.size].reshape(n*m, R)
                V = x[U.size:U.size+V.size].reshape(m*p, R)
                W = x[U.size+V.size:].reshape(p*n, R)
                return compute_error(U, V, W)
            
            # Use L-BFGS-B for optimization
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                x_opt = res.x
                U_opt = x_opt[:U.size].reshape(n*m, R)
                V_opt = x_opt[U.size:U.size+V.size].reshape(m*p, R)
                W_opt = x_opt[U.size+V.size:].reshape(p*n, R)
                
                # Refine to integers
                U_int, V_int, W_int, success = refine_integer(U_opt, V_opt, W_int)
                
                if success:
                    return U_int, V_int, W_int
            except Exception:
                continue
                
        return None

    # Naive solution (Rank 24)
    best_rank = n * m * p
    U_naive = np.zeros((n * m, best_rank), dtype=float)
    V_naive = np.zeros((m * p, best_rank), dtype=float)
    W_naive = np.zeros((p * n, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j, col] = 1.0
                V_naive[j * p + k, col] = 1.0
                W_naive[k * n + i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U_naive.tolist(),
        "V": V_naive.tolist(),
        "W": W_naive.tolist()
    }
    
    # Try to find lower rank
    # Known lower bound is ~19. We try 19, 20, 21, 22, 23
    for R in range(19, best_rank):
        if time.time() > deadline - 30:
            break
            
        res = try_rank_numeric(R)
        if res is not None:
            U_found, V_found, W_found = res
            # Final verification
            if compute_error(U_found, V_found, W_found) < 1e-9:
                best_result = {
                    "U": U_found.tolist(),
                    "V": V_found.tolist(),
                    "W": W_found.tolist()
                }
                best_rank = R
                break # Found minimal rank, stop
        
    return best_result
