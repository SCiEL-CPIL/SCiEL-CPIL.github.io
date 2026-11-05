import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    def make_tensor(n, m, p):
        """Create the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1"""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = make_tensor(n, m, p)
    
    def verify_fastest(U, V, W, T, atol=1e-6):
        """Fast verification using tensor contraction via einsum."""
        if U.size == 0 or V.size == 0 or W.size == 0:
            return False
        rank = U.shape[1]
        if rank == 0:
            return np.allclose(T, 0, atol=atol)
        computed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(computed, T, atol=atol)
    
    def schoolbook(n, m, p):
        """Return the schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def random_decomposition(rank, seed=None):
        """Generate a random decomposition and try to fit it to T using ALS."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid overflow
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        # Scale to match the magnitude of T
        # T has entries 0 or 1, so the decomposition should sum to ~1
        U *= (np.linalg.norm(T) ** (1/3))
        V *= (np.linalg.norm(T) ** (1/3))
        W *= (np.linalg.norm(T) ** (1/3))
        
        return U, V, W
    
    def als_solve(T, n, m, p, U, V, W, max_iters=100, tol=1e-8, verbose=False):
        """
        Alternating Least Squares to optimize U, V, W to decompose T.
        T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        """
        rank = U.shape[1]
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Update U: for each i, solve for U[i,:]
            # T[i,:,:] = sum_r U[i,r] * outer(V[:,r], W[:,r])
            # Reshape: T_i = sum_r U[i,r] * vec(outer(V_r, W_r))
            # So: A @ U[i,:] = T_i.flatten()
            # where A = [vec(outer(V_0, W_0)), vec(outer(V_1, W_1)), ...]
            
            # Precompute A for U update
            # A has shape (m*p * p*n, rank)
            A_U = np.zeros((m*p * p*n, rank))
            for r in range(rank):
                A_U[:, r] = np.outer(V_curr[:, r], W_curr[:, r]).flatten()
            
            for i in range(n*m):
                b = T[i, :, :].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U_curr[i, :] = sol
                except:
                    pass
            
            # Update V: for each j, solve for V[j,:]
            # T[:,j,:] = sum_r V[j,r] * outer(U[:,r], W[:,r])
            A_V = np.zeros((n*m * p*n, rank))
            for r in range(rank):
                A_V[:, r] = np.outer(U_curr[:, r], W_curr[:, r]).flatten()
            
            for j in range(m*p):
                b = T[:, j, :].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V_curr[j, :] = sol
                except:
                    pass
            
            # Update W: for each k, solve for W[k,:]
            # T[:,:,k] = sum_r W[k,r] * outer(U[:,r], V[:,r])
            A_W = np.zeros((n*m * m*p, rank))
            for r in range(rank):
                A_W[:, r] = np.outer(U_curr[:, r], V_curr[:, r]).flatten()
            
            for k in range(p*n):
                b = T[:, :, k].flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W_curr[k, :] = sol
                except:
                    pass
            
            # Check convergence
            if it > 0:
                computed = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                err = np.max(np.abs(computed - T))
                if verbose and it % 10 == 0:
                    print(f"  ALS iter {it}: err = {err:.2e}")
                if err < tol:
                    break
        
        computed = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
        err = np.max(np.abs(computed - T))
        return U_curr, V_curr, W_curr, err
    
    def try_rank(rank, num_trials=5, seed_offset=0):
        """Try to find a valid decomposition of the given rank."""
        best_U, best_V, best_W, best_err = None, None, None, np.inf
        
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            
            seed = seed_offset + trial * 1000
            U, V, W = random_decomposition(rank, seed=seed)
            
            # Run ALS
            U_sol, V_sol, W_sol, err = als_solve(T, n, m, p, U, V, W, max_iters=200, tol=1e-8)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_sol, V_sol, W_sol
            
            if best_err < 1e-6:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Start with schoolbook
    U_school, V_school, W_school = schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Theoretical lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # Also rank >= ceil(n*m*p / max(n,m,p)) = ceil(72/8) = 9 (weak)
    # Strassen-like bounds suggest rank could be significantly lower than 72
    
    # Try various ranks from low to high
    # We'll try a range of ranks and see what works
    ranks_to_try = []
    # Start from a reasonable lower bound and go up
    min_possible = max(n*m, m*p, p*n)  # 24
    # Try ranks from min_possible to schoolbook rank
    for r in range(min_possible, best_rank):
        ranks_to_try.append(r)
    
    # Sort by likelihood of success (try intermediate ranks first)
    # Actually, let's try a strategic order:
    # 1. Try some lower ranks
    # 2. If none work, try higher ranks
    
    # Let's try ranks in a specific order
    trial_ranks = []
    # Try every 2nd rank from min to max
    for r in range(min_possible, best_rank, 2):
        trial_ranks.append(r)
    # Make sure we include some key ranks
    for r in [24, 25, 26, 27, 28, 30, 32, 36, 40, 48, 54, 60, 64, 68, 70, 71]:
        if min_possible <= r < best_rank and r not in trial_ranks:
            trial_ranks.append(r)
    
    trial_ranks = sorted(set(trial_ranks))
    
    for rank in trial_ranks:
        if time.time() > deadline - 15.0:
            break
        
        if rank >= best_rank:
            break
        
        # Number of trials depends on time budget
        time_per_trial = 5.0  # seconds per trial estimate
        time_remaining = deadline - 15.0 - time.time()
        num_trials = max(1, int(time_remaining / time_per_trial))
        num_trials = min(num_trials, 20)  # Cap at 20 trials per rank
        
        U_try, V_try, W_try, err = try_rank(rank, num_trials=num_trials, seed_offset=rank * 1000)
        
        if U_try is not None and err < 1e-6:
            # Verify the solution
            if verify_fastest(U_try, V_try, W_try, T):
                best_rank = rank
                best_U, best_V, best_W = U_try, V_try, W_try
                print(f"  Found valid decomposition with rank {rank}!")
                # We found a valid one, but let's try to go lower
                # Continue searching for even lower ranks
    
    # Final verification
    if best_U is not None:
        if not verify_fastest(best_U, best_V, best_W, T):
            # Fall back to schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = n * m * p
    
    # Remove zero columns to minimize rank
    def remove_zero_cols(U, V, W, tol=1e-9):
        rank = U.shape[1]
        keep = []
        for r in range(rank):
            if np.max(np.abs(U[:, r])) > tol or \
               np.max(np.abs(V[:, r])) > tol or \
               np.max(np.abs(W[:, r])) > tol:
                keep.append(r)
        
        if len(keep) == rank:
            return U, V, W, rank
        
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, len(keep)
    
    U_final, V_final, W_final, final_rank = remove_zero_cols(best_U, best_V, best_W)
    
    # Verify the final result
    if not verify_fastest(U_final, V_final, W_final, T):
        # Fall back to schoolbook
        U_final, V_final, W_final = U_school, V_school, W_school
        final_rank = n * m * p
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
