"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U0 = np.zeros((n * m, rank_school))
    V0 = np.zeros((m * p, rank_school))
    W0 = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s limit
    
    best_rank = rank_school
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Precompute index mappings for efficiency
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k  
    # W indices: (k, i) -> k*n + i
    
    # For each target rank, try to find a decomposition
    # The lower bound is max(n*m, m*p, p*n) = 25
    # But let's try from a reasonable range
    
    # Use vectorized ALS for speed
    def compute_error(U, V, W):
        """Compute the error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                    # Actually, it's sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return np.sqrt(err)
    
    def als_solve(U, V, W, target_rank, max_iters=50, tol=1e-8):
        """Alternating least squares for tensor decomposition."""
        U_curr = U.copy()
        V_curr = V.copy() 
        W_curr = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U_curr)
            for i in range(n):
                for j in range(m):
                    # A[c,c'] = sum_k V[j*p+k,c] * W[k*n+i,c']
                    A_mat = np.zeros((target_rank, target_rank))
                    for k in range(p):
                        A_mat += np.outer(V_curr[j*p+k, :], W_curr[k*n+i, :])
                    try:
                        U_new[i*m+j, :] = np.linalg.solve(A_mat + 1e-10*np.eye(target_rank), np.ones(target_rank))
                    except:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_mat + 1e-10*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
            U_curr = U_new
            
            # Update V
            V_new = np.zeros_like(V_curr)
            for j in range(m):
                for k in range(p):
                    # A[c,c'] = sum_i U[i*m+j,c] * W[k*n+i,c']
                    A_mat = np.zeros((target_rank, target_rank))
                    for i in range(n):
                        A_mat += np.outer(U_curr[i*m+j, :], W_curr[k*n+i, :])
                    try:
                        V_new[j*p+k, :] = np.linalg.solve(A_mat + 1e-10*np.eye(target_rank), np.ones(target_rank))
                    except:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_mat + 1e-10*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
            V_curr = V_new
            
            # Update W
            W_new = np.zeros_like(W_curr)
            for k in range(p):
                for i in range(n):
                    # A[c,c'] = sum_j U[i*m+j,c] * V[j*p+k,c']
                    A_mat = np.zeros((target_rank, target_rank))
                    for j in range(m):
                        A_mat += np.outer(U_curr[i*m+j, :], V_curr[j*p+k, :])
                    try:
                        W_new[k*n+i, :] = np.linalg.solve(A_mat + 1e-10*np.eye(target_rank), np.ones(target_rank))
                    except:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_mat + 1e-10*np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
            W_curr = W_new
            
            # Check convergence
            err = compute_error(U_curr, V_curr, W_curr)
            if err < tol:
                return U_curr, V_curr, W_curr, True
        
        return U_curr, V_curr, W_curr, False
    
    # Try ranks from a reasonable upper bound down to the lower bound
    # For 3x5x5, let's try ranks from 60 down to 25
    # But start with a few promising ranks
    
    # First, let's try to see if rank 25 is achievable (theoretical lower bound)
    # Then work up
    
    target_ranks = [25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 55, 60, 65, 70, 74]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        num_attempts = 5
        found = False
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 12345 + target_rank * 678)
            U_init = np.random.randn(n * m, target_rank) * 0.3
            V_init = np.random.randn(m * p, target_rank) * 0.3
            W_init = np.random.randn(p * n, target_rank) * 0.3
            
            U_res, V_res, W_res, converged = als_solve(U_init, V_init, W_init, target_rank)
            
            if converged:
                # Verify the solution
                err = compute_error(U_res, V_res, W_res)
                if err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_res.copy(), V_res.copy(), W_res.copy()
                    found = True
                    break
        
        if found:
            break
    
    # Return the best solution found
    if best_rank < rank_school:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U0.tolist()
        V_out = V0.tolist()
        W_out = W0.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
