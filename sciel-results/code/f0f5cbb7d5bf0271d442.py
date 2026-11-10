import numpy as np
import time
import scipy.optimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Initial guess based on Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Expand to 8x8 grid
    # x_ij depends on i, y_ij depends on j in the separable case
    # We will optimize the full 128 parameters
    # params = [x_00, y_00, x_01, y_01, ..., x_77, y_77]
    
    def init_params():
        p = np.empty(N*N*2)
        for i in range(N):
            for j in range(N):
                idx = i*N + j
                p[2*idx] = keich[i]
                p[2*idx+1] = keich[j]
        return p

    p0 = init_params()

    def objective(params):
        # Unpack params
        x = params[0::2].reshape(N, N)
        y = params[1::2].reshape(N, N)
        
        # Calculate bounds
        # X0 = min x_ij
        # X1 = max (x_ij + (i+1)/8)
        # Y0 = min y_ij
        # Y1 = max (y_ij + (j+1)/8)
        
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        
        X0 = np.min(x)
        X1 = np.max(x + (i_grid + 1) / N)
        Y0 = np.min(y)
        Y1 = np.max(y + (j_grid + 1) / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10
            
        # Monte Carlo estimation
        # Use a fixed seed for consistent noise during optimization? 
        # Or just use np.random? 
        # Using a fixed seed for the *optimization* loop might be better for stability, 
        # but the evaluator uses its own seed. 
        # We want to minimize the expected volume, so fixed seed is fine for optimization.
        rng = np.random.default_rng(42)
        M = 20000 # Reduce points for speed during optimization
        a = rng.uniform(X0, X1, M)
        b = rng.uniform(Y0, Y1, M)
        c = rng.uniform(0, 1, M)
        
        # Vectorized check
        # For each tube k (flattened i, j), calculate the interval at c
        # i_flat, j_flat are (N*N,)
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        x_flat = x.flatten()
        y_flat = y.flatten()
        
        # x_int_start = x_k + c * i_k / N
        # x_int_end = x_k + c * i_k / N + 1/N
        # y_int_start = y_k + c * j_k / N
        # y_int_end = y_k + c * j_k / N + 1/N
        
        # Shape: (M, N*N)
        c_reshaped = c[:, np.newaxis]
        i_term = c_reshaped * i_flat[np.newaxis, :] / N
        j_term = c_reshaped * j_flat[np.newaxis, :] / N
        
        x_start = x_flat[np.newaxis, :] + i_term
        x_end = x_start + 1.0/N
        y_start = y_flat[np.newaxis, :] + j_term
        y_end = y_start + 1.0/N
        
        # Check if a is in [x_start, x_end]
        # a shape (M,), x_start shape (M, N*N)
        a_reshaped = a[:, np.newaxis]
        b_reshaped = b[:, np.newaxis]
        
        in_x = (a_reshaped >= x_start) & (a_reshaped <= x_end)
        in_y = (b_reshaped >= y_start) & (b_reshaped <= y_end)
        
        in_tube = in_x & in_y
        
        # If point is in any tube
        hit = np.any(in_tube, axis=1)
        
        fraction = np.sum(hit) / M
        volume = fraction * area
        return volume

    # Optimize
    start_time = time.time()
    timeout = 500 # Leave 100s margin
    
    best_params = p0
    best_score = objective(p0)
    
    # Try Nelder-Mead
    # maxiter is not a strict time bound, so we need to check time in a callback or wrapper
    # Scipy minimize doesn't have a built-in timeout.
    # We can wrap the objective to check time and raise an exception or return a large value.
    # Better: run minimize in a loop with increasing maxiter? 
    # Or just run one long minimize and hope it finishes? 
    # Nelder-Mead can be slow.
    # Let's try a few restarts with Nelder-Mead.
    
    def time_limited_objective(params):
        if time.time() - start_time > timeout:
            raise TimeoutError
        return objective(params)
        
    # Since we can't easily interrupt scipy, we will run it and catch exceptions if we implement a timeout mechanism in the objective?
    # No, scipy will keep calling it.
    # We can use a simple loop:
    # Run Nelder-Mead with a fixed number of iterations.
    
    # Let's try 5 restarts
    for restart in range(5):
        if time.time() - start_time > timeout:
            break
            
        if restart == 0:
            p_start = p0
        else:
            # Perturb best_params
            p_start = best_params + np.random.normal(0, 0.1, size=best_params.shape)
            
        try:
            result = scipy.optimize.minimize(
                time_limited_objective, 
                p_start, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except TimeoutError:
            break
        except Exception:
            pass

    # Final check
    if time.time() - start_time > timeout:
         # Return best known
         pass

    # Unpack best_params
    x_best = best_params[0::2].reshape(N, N)
    y_best = best_params[1::2].reshape(N, N)
    
    # Construct output array
    positions = np.empty((N*N, 2))
    for i in range(N):
        for j in range(N):
            k = i*N + j
            positions[k, 0] = x_best[i, j]
            positions[k, 1] = y_best[i, j]
            
    return positions
