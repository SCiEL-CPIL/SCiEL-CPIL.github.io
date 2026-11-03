"""Tensor decomposition for 3x3x7 matrix multiplication."""

import numpy as np
from scipy.optimize import least_squares
import time

# EVOLVE-BLOCK-START
def construct():
    # Dimensions: A is 3x3, B is 3x7, C is 3x7
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # W indexed as W[k*n + i, r] per trace convention
    
    n, m, p = 3, 3, 7
    dim_u = n * m  # 9
    dim_v = m * p  # 21
    dim_w = p * n  # 21
    
    # Create the target tensor (flattened)
    # T[i, j, k] = 1 if term (i, j, k) exists
    # In the decomposition T = sum U_r[i] V_r[j] W_r[k]
    # The naive decomposition has rank 63.
    # We construct the target tensor T_naive of shape (9, 21, 21)
    T_naive = np.zeros((dim_u, dim_v, dim_w))
    
    # Fill T_naive with the standard multiplication tensor
    # C_ik = sum_j A_ij B_jk
    # T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T_naive[u_idx, v_idx, w_idx] = 1.0

    T_flat = T_naive.flatten()
    
    best_result = None
    best_rank = dim_u * dim_v * dim_w // max(dim_u, dim_v, dim_w) # Just a high number, actually 63
    best_rank = 63
    
    # Fallback: Naive decomposition
    U_naive = np.zeros((dim_u, 63))
    V_naive = np.zeros((dim_v, 63))
    W_naive = np.zeros((dim_w, 63))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, t] = 1.0
                V_naive[j*p+k, t] = 1.0
                W_naive[k*n+i, t] = 1.0
                t += 1
                
    # Try to find a lower rank decomposition
    # We search for ranks R from 45 to 62
    # Use random restarts
    
    def residuals(params, R):
        # params shape: R * (dim_u + dim_v + dim_w)
        U = params[:R*dim_u].reshape(R, dim_u).T
        V = params[R*dim_u:R*(dim_u+dim_v)].reshape(R, dim_v).T
        W = params[R*(dim_u+dim_v):].reshape(R, dim_w).T
        
        # Compute tensor product sum
        # T[r] = U[:, r] outer V[:, r] outer W[:, r]
        # Sum over r
        # Efficiently: einsum
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return (T_recon.flatten() - T_flat)

    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Try ranks
    for R in range(45, 63):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale: values are typically -1, 0, 1. 
            # Init with small random values
            np.random.seed(42 + R + restart)
            params0 = np.random.uniform(-1, 1, size=R * (dim_u + dim_v + dim_w))
            
            try:
                sol = least_squares(residuals, params0, args=(R,), method='lm', max_nfev=1000)
                if sol.success and sol.cost < 1e-10:
                    # Check if we can round to half-integers
                    U_sol = sol.x[:R*dim_u].reshape(R, dim_u).T
                    V_sol = sol.x[R*dim_u:R*(dim_u+dim_v)].reshape(R, dim_v).T
                    W_sol = sol.x[R*(dim_u+dim_v):].reshape(R, dim_w).T
                    
                    # Round to nearest 0.5
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_sol * 2) / 2
                    
                    # Verify exactness of rounded solution
                    T_check = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
                    if np.allclose(T_check.flatten(), T_flat, atol=1e-9):
                        # Found a valid integer/half-integer solution
                        # Check if rank is actually R (some terms might be zero)
                        # Remove zero terms to get minimal rank
                        # A term r is zero if U[:,r]==0 or V[:,r]==0 or W[:,r]==0
                        mask = (np.linalg.norm(U_round, axis=0) > 1e-9) & \
                               (np.linalg.norm(V_round, axis=0) > 1e-9) & \
                               (np.linalg.norm(W_round, axis=0) > 1e-9)
                        
                        U_min = U_round[:, mask]
                        V_min = V_round[:, mask]
                        W_min = W_round[:, mask]
                        
                        actual_rank = U_min.shape[1]
                        
                        if actual_rank < best_rank:
                            best_rank = actual_rank
                            # Convert to lists of lists as required by interface?
                            # The interface expects {'U': ..., 'V': ..., 'W': ...}
                            # The seed program returns lists of lists.
                            # But numpy arrays are often accepted if they convert.
                            # Let's check the seed.
                            # Seed: return {"U": U, "V": V, "W": W} where U is list of lists.
                            # I will return lists of lists to be safe.
                            best_result = {
                                "U": U_min.tolist(),
                                "V": V_min.tolist(),
                                "W": W_min.tolist()
                            }
                            # We found a better rank, can we stop?
                            # Maybe try lower ranks?
                            # The loop goes 45 to 62.
                            # If we find 55, we should continue to 45?
                            # The loop is ascending. So we find 62 first?
                            # No, range(45, 63) goes 45, 46...
                            # So we find the lowest first.
                            # If we find a valid one at 45, that's great.
                            # But it might be harder to find.
                            # If we find one at 55, we should keep looking for 45?
                            # Yes, the loop continues.
                            
            except Exception:
                pass

    if best_result is not None:
        return best_result
    else:
        # Return naive
        return {
            "U": U_naive.tolist(),
            "V": V_naive.tolist(),
            "W": W_naive.tolist()
        }
# EVOLVE-BLOCK-END
