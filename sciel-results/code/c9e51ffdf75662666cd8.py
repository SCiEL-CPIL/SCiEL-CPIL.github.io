import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 12, 24, 18
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    deadline = time.time() + 140  # Safety margin
    
    def reconstruct(U, V, W):
        """Reconstruct tensor from factor matrices."""
        r = U.shape[1]
        T_hat = np.zeros((nm, mp, pn), dtype=np.float64)
        for r_ in range(r):
            T_hat += np.outer(np.outer(U[:, r_], V[:, r_]), W[:, r_]).reshape(nm, mp, pn)
        return T_hat
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        T_hat = reconstruct(U, V, W)
        return np.max(np.abs(T - T_hat)) < tol
    
    def round_to_half_int(X):
        """Round matrix to nearest half-integers."""
        return np.round(X * 2.0) / 2.0
    
    # Schoolbook decomposition (rank 72)
    rank_sb = n * m * p  # 72
    U_sb = np.zeros((nm, rank_sb), dtype=np.float64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.float64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, idx] = 1.0
                V_sb[j*p + k, idx] = 1.0
                W_sb[k*n + i, idx] = 1.0
                idx += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find a 71-term decomposition
    target_rank = 71
    if time.time() < deadline - 5:
        # Initialize from schoolbook, drop one column and perturb
        # Drop the last column (corresponding to i=2, j=3, k=5)
        U0 = best_U[:, :target_rank].copy()
        V0 = best_V[:, :target_rank].copy()
        W0 = best_W[:, :target_rank].copy()
        
        # Add small random perturbation to escape local minima
        np.random.seed(42)
        scale = 0.05
        U0 += np.random.randn(nm, target_rank) * scale
        V0 += np.random.randn(mp, target_rank) * scale
        W0 += np.random.randn(pn, target_rank) * scale
        
        # Use gradient descent / L-BFGS to minimize ||T - T_hat||_F^2
        from scipy.optimize import minimize
        
        def objective(params):
            U_f = params[:nm*target_rank].reshape(nm, target_rank)
            V_f = params[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
            W_f = params[2*nm*target_rank:].reshape(pn, target_rank)
            T_hat = reconstruct(U_f, V_f, W_f)
            R = T - T_hat
            return np.sum(R * R)
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
            x_opt = res.x
            U_opt = x_opt[:nm*target_rank].reshape(nm, target_rank)
            V_opt = x_opt[nm*target_rank:2*nm*target_rank].reshape(mp, target_rank)
            W_opt = x_opt[2*nm*target_rank:].reshape(pn, target_rank)
            
            # Round to half-integers
            U_r = round_to_half_int(U_opt)
            V_r = round_to_half_int(V_opt)
            W_r = round_to_half_int(W_opt)
            
            if check_exact(U_r, V_r, W_r, tol=1e-8):
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
            else:
                # Check if the unrounded solution is close enough
                if check_exact(U_opt, V_opt, W_opt, tol=1e-10):
                    # Try rounding with different strategy
                    for factor in [1, 2, 4, 8]:
                        U_r2 = np.round(U_opt * factor) / factor
                        V_r2 = np.round(V_opt * factor) / factor
                        W_r2 = np.round(W_opt * factor) / factor
                        if check_exact(U_r2, V_r2, W_r2, tol=1e-8):
                            best_rank = target_rank
                            best_U = U_r2
                            best_V = V_r2
                            best_W = W_r2
                            break
        except Exception:
            pass
    
    # If we still have rank 72, try a constructive algebraic approach
    if best_rank == 72 and time.time() < deadline - 10:
        # Constructive 71-term decomposition for 3x4x6
        # Use the following explicit construction based on rank reduction
        # For the last k-index (k=5), we can merge terms.
        
        # The idea: C_{i,5} = sum_j A_{i,j} B_{j,5}
        # We can write B_{j,5} = B_{j,5} - B_{j,4} + B_{j,4}
        # So C_{i,5} = sum_j A_{i,j}(B_{j,5} - B_{j,4}) + sum_j A_{i,j} B_{j,4}
        # This doesn't reduce rank directly, but combined with other tricks...
        
        # Alternative: use a known 71-term construction
        # Based on the fact that the matmul tensor has a 71-term decomposition
        # when one dimension allows a "diagonal" trick.
        
        # Let me try a direct constructive approach:
        # Terms for k=0,1,2,3,4: standard (3*4*5 = 60 terms)
        # For k=5: use 11 terms instead of 12
        
        # C_{i,5} = sum_j A_{i,j} B_{j,5}
        # We can compute this using 11 products by noting that
        # we can share some intermediate values.
        
        # Actually, for a single row of C (i fixed), C_{i,5} = A_{i,0}B_{0,5} + A_{i,1}B_{1,5} + A_{i,2}B_{2,5} + A_{i,3}B_{3,5}
        # This requires 4 products per i, so 12 total for k=5.
        # We can't reduce this for a single entry.
        
        # The rank reduction must come from a global structure.
        # Let me use a different approach: use the "laser method" or
        # a known explicit construction.
        
        # For now, if we can't find 71, return 72.
        pass
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
