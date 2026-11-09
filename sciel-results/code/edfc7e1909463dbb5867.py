"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Define the residual function for a given rank
    def residuals(params, r):
        # Unpack params into U, V, W
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape((n * m, r))
        V = V_flat.reshape((m * p, r))
        W = W_flat.reshape((p * n, r))
        
        # Compute tensor entries
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i, j, k] == 1 for all i, j, k
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    val = np.dot(u_val, v_val * w_val)
                    res[idx] = val - 1.0
                    idx += 1
        return res

    # Try to find lower rank decompositions
    np.random.seed(42)
    
    # Theoretical lower bound is n*p = 14
    # We try ranks from 15 to 41
    for trial_rank in range(15, 42):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            r = trial_rank
            U_init = np.random.randn(n * m, r)
            V_init = np.random.randn(m * p, r)
            W_init = np.random.randn(p * n, r)
            
            # Scale to help convergence?
            # The product U*V*W should be ~1. If U,V,W ~ N(0,1), product ~ N(0, r).
            # Let's scale by 1/sqrt(r)
            scale = 1.0 / np.sqrt(r)
            U_init *= scale
            V_init *= scale
            W_init *= scale
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                # Use least_squares
                result = opt.least_squares(
                    residuals, 
                    x0, 
                    args=(r,), 
                    method='lm', # Levenberg-Marquardt, good for least squares
                    max_nfev=1000
                )
                
                if result.success or result.cost < 1e-6:
                    # Check if it's a valid decomposition
                    final_res = residuals(result.x, r)
                    if np.allclose(final_res, 0, atol=1e-5):
                        # Extract U, V, W
                        size_U = n * m * r
                        size_V = m * p * r
                        U_sol = result.x[:size_U].reshape((n * m, r))
                        V_sol = result.x[size_U:size_U + size_V].reshape((m * p, r))
                        W_sol = result.x[size_U + size_V:].reshape((p * n, r))
                        
                        # Round to nearest integer if close?
                        # The problem asks for integer/half-integer entries.
                        # If the solution is close to integers, we can snap them.
                        U_rounded = np.round(U_sol * 2) / 2
                        V_rounded = np.round(V_sol * 2) / 2
                        W_rounded = np.round(W_sol * 2) / 2
                        
                        # Verify rounded solution
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                                    if abs(val - 1) > 1e-5:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        
                        if valid:
                            best_rank = r
                            best_U = U_rounded.tolist()
                            best_V = V_rounded.tolist()
                            best_W = W_rounded.tolist()
                            found = True
                            break # Move to next lower rank? No, we are going up.
                            # Wait, we are iterating 15..41. If we find rank 20, we should try 15..19 first?
                            # The loop is 15 to 41. So it finds the lowest rank first.
                            # If found, we can break the outer loop?
                            # Yes, we want the minimal rank.
                            # So if we find a valid rank r, we can stop searching for higher ranks?
                            # No, we are searching from low to high. If we find rank 20, 
                            # we have the minimal rank (assuming we didn't miss 15-19).
                            # So we can break the outer loop.
                            
            except Exception:
                pass
        
        if found:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
