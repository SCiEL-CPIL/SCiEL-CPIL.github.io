import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    def compute_residual_fast(U_r, V_r, W_r):
        """Compute residual efficiently using tensor operations"""
        # Compute sum over rank dimension for each (i,j,k)
        # U_r[i*m+j, :], V_r[j*p+k, :], W_r[k*n+i, :]
        res = np.zeros((n, m, p))
        for r in range(U_r.shape[1]):
            u_col = U_r[:, r]
            v_col = V_r[:, r]
            w_col = W_r[:, r]
            # Reshape to get the tensor contribution
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            # The contribution to (i,j,k) is u[i,j] * v[j,k] * w[k,i]
            # This is a rank-1 tensor in (i,j,k) space
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
        return res - 1.0
    
    def residual_vector(x, R):
        """Flattened residual for optimizer"""
        u_size = nm * R
        v_size = mp * R
        
        U_r = x[:u_size].reshape(nm, R)
        V_r = x[u_size:u_size+v_size].reshape(mp, R)
        W_r = x[u_size+v_size:].reshape(pn, R)
        
        res = compute_residual_fast(U_r, V_r, W_r)
        return res.flatten()
    
    def verify_exact(U_r, V_r, W_r, tol=1e-9):
        """Verify that the decomposition is exact (within tolerance)"""
        res = compute_residual_fast(U_r, V_r, W_r)
        return np.max(np.abs(res)) < tol
    
    def polish_to_half_integer(U_sol, V_sol, W_sol, R):
        """Polish a solution to exact half-integer values"""
        # Try rounding to nearest half-integer
        U_round = np.round(U_sol * 2) / 2
        V_round = np.round(V_sol * 2) / 2
        W_round = np.round(W_sol * 2) / 2
        
        if verify_exact(U_round, V_round, W_round):
            return U_round, V_round, W_round
        
        # Try rounding to nearest integer
        U_round2 = np.round(U_sol)
        V_round2 = np.round(V_sol)
        W_round2 = np.round(W_sol)
        
        if verify_exact(U_round2, V_round2, W_round2):
            return U_round2, V_round2, W_round2
        
        # Try a local search around the rounded values
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for delta_u in [-0.5, 0, 0.5]:
            for delta_v in [-0.5, 0, 0.5]:
                for delta_w in [-0.5, 0, 0.5]:
                    U_try = U_round + delta_u
                    V_try = V_round + delta_v
                    W_try = W_round + delta_w
                    
                    # Only consider if values are in a reasonable range
                    if np.max(np.abs(U_try)) > 10 or np.max(np.abs(V_try)) > 10 or np.max(np.abs(W_try)) > 10:
                        continue
                    
                    res = compute_residual_fast(U_try, V_try, W_try)
                    err = np.max(np.abs(res))
                    
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    
                    if err < 1e-9:
                        return U_try, V_try, W_try
        
        return None
    
    def try_reduce_rank(R, max_restarts=20, time_budget=30):
        """Try to find a rank-R decomposition"""
        start_time = time.time()
        
        # Initialize with schoolbook pattern (first R columns)
        U_init = np.zeros((nm, R))
        V_init = np.zeros((mp, R))
        W_init = np.zeros((pn, R))
        
        # Copy first R columns from schoolbook
        U_init[:, :min(R, rank)] = U[:, :min(R, rank)]
        V_init[:, :min(R, rank)] = V[:, :min(R, rank)]
        W_init[:, :min(R, rank)] = W[:, :min(R, rank)]
        
        best_result = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break
            
            # Create initial guess
            if restart == 0:
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Random initialization with some structure
                U_rand = np.random.normal(0, 1, (nm, R))
                V_rand = np.random.normal(0, 1, (mp, R))
                W_rand = np.random.normal(0, 1, (pn, R))
                
                # Scale to reasonable magnitudes
                U_rand /= np.sqrt(R)
                V_rand /= np.sqrt(R)
                W_rand /= np.sqrt(R)
                
                x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            try:
                # Phase 1: Find approximate solution
                sol = least_squares(residual_vector, x0, args=(R,), method='trf', 
                                   max_nfev=2000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                
                if not sol.success:
                    continue
                
                # Check if we have a good approximate solution
                final_err = np.max(np.abs(sol.fun))
                if final_err > 1e-6:
                    continue
                
                # Extract solution
                u_size = nm * R
                v_size = mp * R
                U_sol = sol.x[:u_size].reshape(nm, R)
                V_sol = sol.x[u_size:u_size+v_size].reshape(mp, R)
                W_sol = sol.x[u_size+v_size:].reshape(pn, R)
                
                # Phase 2: Polish to exact half-integers
                polished = polish_to_half_integer(U_sol, V_sol, W_sol, R)
                if polished is not None:
                    best_result = polished
                    break  # Found exact solution, stop searching
            
            except Exception as e:
                continue
        
        return best_result
    
    # Try to reduce rank
    start_time = time.time()
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try ranks from 44 down to 40
    for target_rank in range(rank - 1, max(35, rank - 10), -1):
        elapsed = time.time() - start_time
        if elapsed > 100:  # Safety margin
            break
        
        result = try_reduce_rank(target_rank, max_restarts=15, time_budget=8)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
