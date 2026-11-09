import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Target Tensor T
    # T is indexed by (i, j, k) -> output index (k, i)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[(i,j), (j,k), (k,i)] = 1
    
    # We will work with flattened indices for U, V, W
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)
    
    # Precompute indices for the non-zero entries of T
    # i in [0, n), j in [0, m), k in [0, p)
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                support.append((u_idx, v_idx, w_idx))
                
    # Function to compute residual squared for given U, V, W
    def residual_sq(U, V, W):
        # U, V, W are 2D arrays of shape (dim, r)
        # We need to compute sum_t U[u,t] V[v,t] W[w,t] for each (u,v,w) in support
        # And also ensure it's 0 for others? 
        # Actually, if it matches the support, does it match the zeros?
        # The tensor is defined to be 0 elsewhere.
        # So we must check all possible (u,v,w) combinations?
        # u in [0, n*m), v in [0, m*p), w in [0, p*n)
        # Total combos: 15 * 25 * 15 = 5625.
        # This is manageable.
        
        # Let's compute the full tensor product sum
        # S[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
        # This is a tensor contraction.
        # S = np.einsum('ut,vt,wt->uvw', U, V, W)
        # This creates an array of size 15x25x15.
        
        S = np.einsum('ut,vt,wt->uvw', U, V, W)
        
        # Create target tensor T_full
        T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for u_idx, v_idx, w_idx in support:
            T_full[u_idx, v_idx, w_idx] = 1.0
            
        diff = S - T_full
        return np.sum(diff * diff)

    # Function to verify exactness and half-integer constraint
    def verify(U, V, W):
        # Check half-integer constraint
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-6):
                return False
            if not np.allclose(np.round(mat * 2), mat * 2):
                 return False
        
        # Check exact decomposition
        S = np.einsum('ut,vt,wt->uvw', U, V, W)
        T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for u_idx, v_idx, w_idx in support:
            T_full[u_idx, v_idx, w_idx] = 1.0
        
        if not np.allclose(S, T_full, atol=1e-6):
            return False
        return True

    # Schoolbook solution
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # Try to find lower rank
    # We will try ranks from best_rank - 1 down to some limit
    # Use random restarts and local optimization
    
    def try_lower_rank(target_rank, U_init, V_init, W_init):
        # Remove (best_rank - target_rank) random columns
        # Actually, we start from a perturbed version of the reduced set
        
        # This function is a placeholder for the search logic
        pass

    # Search loop
    # We will try to reduce rank by 1 step at a time?
    # Or try specific ranks.
    
    # Given time constraints, we will try a few random perturbations of the rank 74 case.
    
    current_U, current_V, current_W = schoolbook()
    
    for target_rank in range(best_rank - 1, best_rank - 10, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(50): # 50 attempts per rank
            if time.time() > deadline - 2:
                break
                
            # Start with schoolbook, remove random columns
            cols = np.arange(best_rank)
            np.random.shuffle(cols)
            keep_cols = cols[:target_rank]
            
            U_try = current_U[:, keep_cols].copy()
            V_try = current_V[:, keep_cols].copy()
            W_try = current_W[:, keep_cols].copy()
            
            # Add small noise to escape local minima
            U_try += np.random.normal(0, 0.1, U_try.shape)
            V_try += np.random.normal(0, 0.1, V_try.shape)
            W_try += np.random.normal(0, 0.1, W_try.shape)
            
            # Use scipy.optimize.minimize
            from scipy.optimize import minimize
            
            # Flatten parameters
            x0 = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
            
            def objective(x):
                U_flat = x[:U_try.size]
                V_flat = x[U_try.size:U_try.size + V_try.size]
                W_flat = x[U_try.size + V_try.size:]
                
                U_curr = U_flat.reshape(U_try.shape)
                V_curr = V_flat.reshape(V_try.shape)
                W_curr = W_flat.reshape(W_try.shape)
                
                return residual_sq(U_curr, V_curr, W_curr)
            
            # We don't need gradients for a quick test, but L-BFGS is better with them.
            # For simplicity and speed, use BFGS or Nelder-Mead?
            # Nelder-Mead is derivative-free and robust for small problems.
            # But 15*74 + 25*74 + 15*74 = 3700 variables. Nelder-Mead is too slow.
            # We need gradients.
            
            def gradient(x):
                U_flat = x[:U_try.size]
                V_flat = x[U_try.size:U_try.size + V_try.size]
                W_flat = x[U_try.size + V_try.size:]
                
                U_curr = U_flat.reshape(U_try.shape)
                V_curr = V_flat.reshape(V_try.shape)
                W_curr = W_flat.reshape(W_try.shape)
                
                S = np.einsum('ut,vt,wt->uvw', U_curr, V_curr, W_curr)
                T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for u_idx, v_idx, w_idx in support:
                    T_full[u_idx, v_idx, w_idx] = 1.0
                
                diff = S - T_full
                
                # dS/du = V[v,t] W[w,t]
                # dL/du = 2 * sum_{v,w} diff[u,v,w] * V[v,t] W[w,t]
                # This is an einsum too.
                
                grad_U = 2 * np.einsum('uvw,vt,wt->ut', diff, V_curr, W_curr)
                grad_V = 2 * np.einsum('uvw,ut,wt->vt', diff, U_curr, W_curr)
                grad_W = 2 * np.einsum('uvw,ut,vt->wt', diff, U_curr, V_curr)
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-6:
                    x_opt = res.x
                    U_opt = x_opt[:U_try.size].reshape(U_try.shape)
                    V_opt = x_opt[U_try.size:U_try.size + V_try.size].reshape(V_try.shape)
                    W_opt = x_opt[U_try.size + V_try.size:].reshape(W_try.shape)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    if verify(U_snap, V_snap, W_snap):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = target_rank
                        break # Found a valid solution for this rank, try lower
            except Exception as e:
                pass

    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
