import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build target tensor (sparse)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                
    # Flatten target for optimization
    target_flat = target.flatten()
    # Indices of non-zero elements for faster residual calculation?
    # least_squares requires a vector residual.
    # We can compute residual for all elements, but it's 2025 floats.
    # This is small enough.
    
    def residual(params):
        # params: U (dim_u * R), V (dim_v * R), W (dim_w * R) flattened
        # We need to know R to reshape.
        # This function is called inside the optimization loop where R is fixed.
        pass
        
    # We will try different ranks
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_sb = np.zeros((dim_u, best_rank))
    V_sb = np.zeros((dim_v, best_rank))
    W_sb = np.zeros((dim_w, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
                
    # Try to find lower rank
    # Time limit 120s.
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    # Try ranks from best_rank - 1 down to 35
    # We will attempt multiple restarts for each rank to increase the chance of finding a solution.
    for R in range(best_rank - 1, 34, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Determine how many restarts we can afford
        # Let's try up to 10 restarts, but limited by time.
        max_restarts = 10
        restarts_done = 0
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize randomly with a different seed for each restart
            seed_val = 42 + R * 100 + restart
            np.random.seed(seed_val)
            U_init = np.random.randn(dim_u * R)
            V_init = np.random.randn(dim_v * R)
            W_init = np.random.randn(dim_w * R)
            
            def res_func(params, R=R):
                U = params[:dim_u*R].reshape(dim_u, R)
                V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                
                # Compute tensor product
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                return (T_recon - target).flatten()
                
            try:
                # Use least_squares
                # Limit function evaluations to control time per restart
                # 500 evals is usually fast enough for small problems
                result = least_squares(
                    res_func, 
                    np.concatenate([U_init, V_init, W_init]), 
                    method='lm', 
                    max_nfev=500
                )
                
                # Check if the cost is very low
                if result.cost < 1e-8:
                    params = result.x
                    U = params[:dim_u*R].reshape(dim_u, R)
                    V = params[dim_u*R:dim_u*R + dim_v*R].reshape(dim_v, R)
                    W = params[dim_u*R + dim_v*R:].reshape(dim_w, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify
                    T_snap = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, target, atol=1e-9):
                        best_rank = R
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        break # Found a valid solution for this R, move to next lower R
            except Exception:
                continue

            
    if best_U is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
