import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 135.0  # Leave 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T
    # T[i, j, j2, k, i2, k2] = 1 if j==j2 and i==i2 and k==k2 else 0
    T = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k, i, k] = 1.0
                
    T_flat = T.flatten()
    
    def residual_fn(params, R):
        size_U = n * m * R
        size_V = m * p * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        # U: (n*m, R) -> reshape to (R, n, m)
        U = U_flat.reshape((n * m, R)).T  # (R, n*m)
        V = V_flat.reshape((m * p, R)).T  # (R, m*p)
        W = W_flat.reshape((p * n, R)).T  # (R, p*n)
        
        U_reshaped = U.reshape((R, n, m))
        V_reshaped = V.reshape((R, m, p))
        W_reshaped = W.reshape((R, p, n))
        
        # Reconstruction: T[i,j,j2,k,i2,k2] = sum_r U[r,i,j] * V[r,j2,k] * W[r,k2,i2]
        # But we need the trace convention: W[k*n+i, r] means W indexed by (k, i)
        # So W_reshaped[r, k, i] where W is (p*n, R) with index k*n+i
        # This means W_reshaped has shape (R, p, n) with indices (r, k, i)
        
        # The einsum: sum over r of U[r,i,j] * V[r,j2,k] * W[r,k2,i2]
        # Output shape: (n, m, m, p, n, p)
        # i from U[0], j from U[1], j2 from V[0], k from V[1], i2 from W[1], k2 from W[0]
        recon = np.einsum('rij,rjk,rki->ijjik', U_reshaped, V_reshaped, W_reshaped)
        
        return (recon.flatten() - T_flat)
    
    def verify_solution(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact"""
        R = U.shape[1]
        U_reshaped = U.T.reshape(R, n, m)
        V_reshaped = V.T.reshape(R, m, p)
        W_reshaped = W.T.reshape(R, p, n)
        
        recon = np.einsum('rij,rjk,rki->ijjik', U_reshaped, V_reshaped, W_reshaped)
        return np.max(np.abs(recon - T)) < tol
    
    def snap_to_half_integers(arr, tol=1e-6):
        """Try to snap values to half-integers"""
        snapped = np.round(arr * 2) / 2
        return snapped
    
    def get_schoolbook_solution():
        """Return the standard schoolbook decomposition with rank = n*m*p"""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        return U, V, W
    
    def try_optimize(R, n_restarts=5, warm_start=None):
        """Try to find a solution of rank R using gradient-based optimization"""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        best_result = None
        best_cost = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            if warm_start is not None and restart == 0:
                x0 = warm_start.copy()
            else:
                # Use structured initialization based on partial schoolbook
                x0 = np.random.randn(total_size) * 0.1
                # Add some structure: initialize with small values that might help convergence
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if i * m * p + j * p + k < total_size:
                                x0[i * m * p + j * p + k] += 0.5
                                idx_u = (i * m + j) * R
                                idx_v = (j * p + k) * R
                                idx_w = (k * n + i) * R
                                if idx_u < size_U:
                                    x0[idx_u] += 0.5
                                if idx_v < size_U + size_V:
                                    x0[size_U + idx_v] += 0.5
                                if idx_w < total_size:
                                    x0[size_U + size_V + idx_w] += 0.5
            
            try:
                # Use scipy's least_squares with LM method
                from scipy.optimize import least_squares
                
                res = least_squares(
                    residual_fn, x0, args=(R,),
                    method='lm', max_nfev=200,
                    xtol=1e-10, ftol=1e-10, gtol=1e-10
                )
                
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_result = res
                    
                    if res.cost < 1e-12:
                        break
                        
            except Exception as e:
                continue
        
        if best_result is None:
            return None
        
        if best_result.cost < 1e-10:
            U_flat = best_result.x[:size_U]
            V_flat = best_result.x[size_U:size_U + size_V]
            W_flat = best_result.x[size_U + size_V:]
            
            U = U_flat.reshape((n * m, R))
            V = V_flat.reshape((m * p, R))
            W = W_flat.reshape((p * n, R))
            
            # Try snapping to half-integers
            U_snap = snap_to_half_integers(U)
            V_snap = snap_to_half_integers(V)
            W_snap = snap_to_half_integers(W)
            
            if verify_solution(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
            
            # If snapping doesn't work, check original
            if verify_solution(U, V, W):
                return U, V, W
        
        return None
    
    # Try to find the minimum rank
    best_solution = None
    best_rank = n * m * p
    
    # Lower bound: max(n*p, m*p, n*m) = max(21, 28, 12) = 28
    lower_bound = max(n * p, m * p, n * m)
    
    # Try ranks from lower_bound up to schoolbook
    # Be conservative with number of attempts due to time budget
    for R in range(lower_bound, min(n * m * p + 1, lower_bound + 20)):
        if time.time() > deadline:
            break
            
        result = try_optimize(R, n_restarts=3)
        
        if result is not None:
            U, V, W = result
            best_rank = R
            best_solution = (U, V, W)
            break  # We found the minimum rank
    
    # If no better solution found, use schoolbook
    if best_solution is None:
        U, V, W = get_schoolbook_solution()
    else:
        U, V, W = best_solution
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
