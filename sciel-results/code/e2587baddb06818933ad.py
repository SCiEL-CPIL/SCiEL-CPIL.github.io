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
    deadline = start_time + 125  # Leave 25s safety margin before 150s limit

    best_rank = rank_school
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        return err

    def als_step(U, V, W, target_rank, tol=1e-12):
        """One full ALS iteration: update U, V, W in sequence."""
        U_new = np.zeros_like(U)
        V_new = np.zeros_like(V)
        W_new = np.zeros_like(W)
        
        # Precompute some structures for speed
        # Update U: for each (i,j), solve A_mat @ U_new[i*m+j,:] = ones
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((target_rank, target_rank))
                for k in range(p):
                    A_mat += np.outer(V[j * p + k, :], W[k * n + i, :])
                try:
                    U_new[i * m + j, :] = np.linalg.solve(A_mat + tol * np.eye(target_rank), np.ones(target_rank))
                except np.linalg.LinAlgError:
                    U_new[i * m + j, :] = np.linalg.lstsq(A_mat + tol * np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
        U = U_new

        # Update V: for each (j,k), solve A_mat @ V_new[j*p+k,:] = ones
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((target_rank, target_rank))
                for i in range(n):
                    A_mat += np.outer(U[i * m + j, :], W[k * n + i, :])
                try:
                    V_new[j * p + k, :] = np.linalg.solve(A_mat + tol * np.eye(target_rank), np.ones(target_rank))
                except np.linalg.LinAlgError:
                    V_new[j * p + k, :] = np.linalg.lstsq(A_mat + tol * np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
        V = V_new

        # Update W: for each (k,i), solve A_mat @ W_new[k*n+i,:] = ones
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((target_rank, target_rank))
                for j in range(m):
                    A_mat += np.outer(U[i * m + j, :], V[j * p + k, :])
                try:
                    W_new[k * n + i, :] = np.linalg.solve(A_mat + tol * np.eye(target_rank), np.ones(target_rank))
                except np.linalg.LinAlgError:
                    W_new[k * n + i, :] = np.linalg.lstsq(A_mat + tol * np.eye(target_rank), np.ones(target_rank), rcond=None)[0]
        W = W_new

        return U, V, W

    def polish(U, V, W, target_rank, max_iter=50, tol=1e-14):
        """High-precision polish pass."""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            U, V, W = als_step(U, V, W, target_rank, tol=tol)
            err = compute_error(U, V, W)
            if err < 1e-12:
                break
        return U, V, W, err

    # Try ranks from just below schoolbook down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = 25
    # Realistically for 3x5x5, minimum rank is probably around 30-40
    lower_bound = max(n * m, m * p, p * n)
    
    # Try a range of target ranks
    # Be strategic: try ranks that are likely to work
    target_ranks_to_try = []
    # Start from a few below schoolbook and go down
    for r in range(rank_school - 1, lower_bound - 1, -1):
        target_ranks_to_try.append(r)
    
    # Also add some specific ranks that might be special
    for r in [45, 40, 38, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if r not in target_ranks_to_try and r >= lower_bound:
            target_ranks_to_try.append(r)
    
    target_ranks_to_try = sorted(set(target_ranks_to_try), reverse=True)
    
    # Limit the number of ranks we try to stay within time budget
    # Each rank with a few attempts should take a few seconds
    max_ranks_to_try = 30
    
    np.random.seed(42)
    
    for target_rank in target_ranks_to_try[:max_ranks_to_try]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_attempts = 4
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Small random
                scale = 0.3
            elif attempt == 1:
                # Medium random
                scale = 1.0
            elif attempt == 2:
                # Large random
                scale = 3.0
            else:
                # Sparse-ish random
                scale = 0.5
            
            np.random.seed(attempt * 10000 + target_rank * 100 + 7)
            U_curr = np.random.randn(n * m, target_rank) * scale
            V_curr = np.random.randn(m * p, target_rank) * scale
            W_curr = np.random.randn(p * n, target_rank) * scale
            
            # Run ALS for a number of iterations
            max_als_iter = 30
            for iteration in range(max_als_iter):
                if time.time() > deadline:
                    break
                
                U_curr, V_curr, W_curr = als_step(U_curr, V_curr, W_curr, target_rank, tol=1e-10)
                
                err = compute_error(U_curr, V_curr, W_curr)
                if err < 1e-10:
                    break
            
            # Check if we converged
            err = compute_error(U_curr, V_curr, W_curr)
            if err < 1e-10:
                # Polish for higher precision
                U_curr, V_curr, W_curr, final_err = polish(U_curr, V_curr, W_curr, target_rank, max_iter=30, tol=1e-14)
                if final_err < 1e-10:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    found = True
                    break
        
        if found:
            # Found a lower rank, continue trying even lower
            continue
    
    # If we found a lower rank, use it; otherwise use schoolbook
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
