import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # Target ranks to try, starting from low
    target_ranks = [14, 15, 16, 17]
    
    deadline = time.time() + 140 # Leave 10s margin
    
    best_result = None
    best_rank = rank_school
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    # Convert to lists for return if needed, but we can keep as arrays and convert at end
    # The interface expects {'U': ..., 'V': ..., 'W': ...}
    
    for R in target_ranks:
        if time.time() > deadline - 20: # Leave time for at least one more try or exit
            break
            
        # Try to find decomposition of rank R
        success = False
        max_restarts = 10
        if R == 14:
            max_restarts = 20 # More effort for the best rank
            
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale initialization to help convergence
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*R]
                V_flat = x[n*m*R : n*m*R + m*p*R]
                W_flat = x[n*m*R + m*p*R : ]
                
                U = U_flat.reshape(n*m, R)
                V = V_flat.reshape(m*p, R)
                W = W_flat.reshape(p*n, R)
                
                # Compute T_ijk
                # T_ijk = sum_r U_ir V_jr W_kr
                # Use einsum
                # U: (n*m, R), V: (m*p, R), W: (p*n, R)
                # We need to reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
                U_reshaped = U.reshape(n, m, R)
                V_reshaped = V.reshape(m, p, R)
                W_reshaped = W.reshape(p, n, R)
                
                # Sum over R
                # T[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                # Note: W index is k*n + i. So W_reshaped[k, i, r].
                
                # Efficient computation:
                # Step 1: UV[i, k, r] = sum_j U[i, j, r] V[j, k, r]
                # U_reshaped: (n, m, R) -> transpose to (n, R, m)
                # V_reshaped: (m, p, R)
                # UV = np.einsum('nir,jkr->nkjr', U_reshaped, V_reshaped) # This is (n, p, R, R) ? No.
                
                # Let's do it step by step.
                # U: (n, m, R)
                # V: (m, p, R)
                # We want sum_j U[i,j,r] V[j,k,s]. This couples r and s if we don't sum r.
                # But we want sum_r U[i,j,r] V[j,k,r] W[k,i,r].
                
                # Let's compute M[i, k, r] = sum_j U[i, j, r] V[j, k, r]
                # U_reshaped.T: (n, R, m)
                # V_reshaped: (m, p, R)
                # M = np.einsum('nrm,jpr->nkpr', U_reshaped, V_reshaped) 
                # Wait, V_reshaped is (m, p, R). So V[j, k, r].
                # M[i, k, r] = sum_j U[i, j, r] V[j, k, r]
                # U_reshaped: (n, m, R)
                # V_reshaped: (m, p, R)
                # M = np.einsum('njr,jkr->nkr', U_reshaped, V_reshaped)
                
                M = np.einsum('njr,jkr->nkr', U_reshaped, V_reshaped)
                
                # Now T[i, k] = sum_r M[i, k, r] W[k, i, r]
                # W_reshaped: (p, n, R) -> W[k, i, r]
                # T = np.einsum('nkr,kir->nk', M, W_reshaped)
                
                T = np.einsum('nkr,kir->nk', M, W_reshaped)
                
                # T is (n, p). We want T[i, k] = 1.
                # Error
                err = np.sum((T - 1.0)**2)
                return err
                
            # Optimization
            # Use BFGS or L-BFGS-B
            # BFGS is good for small problems
            try:
                res = opt.minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-8})
                if res.fun < 1e-6:
                    # Found a solution
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R : ].reshape(p*n, R)
                    
                    # Try to snap to half-integers
                    # Round to nearest 0.5
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify snapped solution
                    # Reconstruct T
                    U_r = U_snap.reshape(n, m, R)
                    V_r = V_snap.reshape(m, p, R)
                    W_r = W_snap.reshape(p, n, R)
                    M_snap = np.einsum('njr,jkr->nkr', U_r, V_r)
                    T_snap = np.einsum('nkr,kir->nk', M_snap, W_r)
                    
                    if np.max(np.abs(T_snap - 1.0)) < 1e-9:
                        # Success
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        success = True
                        break
                    else:
                        # Check if float solution is good enough? 
                        # The prompt asks for integer/half-integer.
                        # If snapping fails, maybe try other rounding strategies or accept float if allowed?
                        # "integer/half-integer entries" usually implies strict.
                        # Let's check if the float solution is very close to half-integers.
                        pass
            except Exception:
                pass
                
        if success:
            break

    if best_rank < rank_school and best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
