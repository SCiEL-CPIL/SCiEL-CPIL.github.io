import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _compute_error(U, V, W, n, m, p):
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    D = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.sum((D - 1.0)**2)

def construct():
    n, m, p = DIMENSIONS
    dim = n * m * p
    deadline = time.time() + 110
    
    # Schoolbook
    U_school = np.zeros((n*m, dim), dtype=np.float64)
    V_school = np.zeros((m*p, dim), dtype=np.float64)
    W_school = np.zeros((p*n, dim), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = dim

    # Try to reduce rank by 1
    # Strategy: For each column c, try to zero it out and optimize the rest.
    
    # We will try this for a few columns, with a time limit per attempt.
    # If successful, we have rank 74.
    # Then we can try to reduce from 74 to 73, etc.
    
    def try_reduce_rank(U_init, V_init, W_init, current_rank, target_rank):
        # current_rank is the rank of the init matrices
        # We want to reach target_rank
        # We will iteratively pick a column to zero out and optimize.
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        r = current_rank
        while r > target_rank:
            # Try to remove one column
            # We try all columns? Or random?
            # Trying all is expensive if optimization is slow.
            # Let's try random columns with a time limit.
            
            success = False
            attempts = 0
            start_time = time.time()
            
            while time.time() < start_time + 2.0: # 2s per removal attempt
                c = np.random.randint(r)
                
                # Save state
                U_save = U.copy()
                V_save = V.copy()
                W_save = W.copy()
                
                # Zero out column c
                U[:, c] = 0.0
                V[:, c] = 0.0
                W[:, c] = 0.0
                
                # Optimize the remaining r-1 columns
                # We use gradient descent on the active columns
                # To save time, we only optimize a few steps.
                
                active_cols = [i for i in range(r) if i != c]
                
                # We need to compute gradients or just do random perturbations?
                # Random perturbations (hill climbing) on the active columns.
                
                best_err = _compute_error(U, V, W, n, m, p)
                
                # Hill climbing for a short time
                opt_start = time.time()
                while time.time() < opt_start + 1.0:
                    # Pick a random active column
                    c_act = np.random.choice(active_cols)
                    
                    # Perturb
                    step = 0.1
                    dU = np.random.randn(n*m) * step
                    dV = np.random.randn(m*p) * step
                    dW = np.random.randn(p*n) * step
                    
                    U_new = U.copy()
                    V_new = V.copy()
                    W_new = W.copy()
                    
                    U_new[:, c_act] += dU
                    V_new[:, c_act] += dV
                    W_new[:, c_act] += dW
                    
                    err_new = _compute_error(U_new, V_new, W_new, n, m, p)
                    
                    if err_new < best_err:
                        U, V, W = U_new, V_new, W_new
                        best_err = err_new
                        
                if best_err < 1e-6:
                    # Success! We found a configuration for rank r-1
                    # We need to clean up: remove the zero column
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    r -= 1
                    success = True
                    break
                else:
                    # Restore state
                    U, V, W = U_save, V_save, W_save
                    attempts += 1
                    if attempts > 10:
                        break
                        
            if not success:
                return None, None, None, r
            # If success, continue to next reduction
        return U, V, W, r

    # Start with rank 75, try to get to 74, then 73...
    # We allocate time for each step.
    
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    
    # Try to reduce to 74
    if time.time() < deadline - 20:
        res = try_reduce_rank(current_U, current_V, current_W, current_rank, 74)
        if res[3] == 74:
            current_U, current_V, current_W, current_rank = res
            if current_rank < best_rank:
                best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                
    # Try to reduce to 73
    if current_rank == 74 and time.time() < deadline - 10:
        res = try_reduce_rank(current_U, current_V, current_W, current_rank, 73)
        if res[3] == 73:
            current_U, current_V, current_W, current_rank = res
            if current_rank < best_rank:
                best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank

    # Round and verify best
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    err = _compute_error(U_out, V_out, W_out, n, m, p)
    if err > 1e-9:
        # Fallback
        U_out = np.round(U_school * 2) / 2
        V_out = np.round(V_school * 2) / 2
        W_out = np.round(W_school * 2) / 2
        best_rank = dim

    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
