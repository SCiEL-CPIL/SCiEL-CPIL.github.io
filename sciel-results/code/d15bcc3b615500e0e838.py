import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_half_integer(x, tol=1e-9):
    """Check if x is within tol of a half-integer"""
    rounded = np.round(x * 2) / 2
    return abs(x - rounded) < tol

def _try_solve_for_W(U, V, n, m, p, R):
    """
    Given U (n*m x R) and V (m*p x R), try to find W (p*n x R) with half-integer entries
    such that for all i,k: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    """
    W_sol = np.zeros((p * n, R))
    
    for i in range(n):
        for k in range(p):
            # Build A matrix: A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Solve A @ w = ones(m)
            # Try to find a half-integer solution
            # First get least squares solution
            try:
                w_ls, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
            except:
                return None
            
            # Check if LS solution is already half-integer
            w_rounded = np.round(w_ls * 2) / 2
            residual = np.max(np.abs(A @ w_rounded - np.ones(m)))
            
            if residual < 1e-9:
                W_sol[k * n + i, :] = w_rounded
            else:
                # Try to find a half-integer solution in the affine space
                # w = w_ls + null_space @ t
                # We need A @ w = ones(m)
                # If rank_A < m, there's a null space
                if rank_A < m:
                    # Get null space
                    if rank_A == 0:
                        # A is all zeros - no solution unless ones(m) is zero (it's not)
                        return None
                    # Compute null space using SVD
                    U_svd, S, Vt = np.linalg.svd(A, full_matrices=True)
                    null_space = Vt[rank_A:, :].T  # (R x (R-rank_A))
                    
                    # Try random t values in null space to find half-integer solution
                    found = False
                    for attempt in range(200):
                        t = np.random.randint(-4, 5, size=(R - rank_A,)) * 0.5
                        w_candidate = w_ls + null_space @ t
                        w_candidate_rounded = np.round(w_candidate * 2) / 2
                        res = np.max(np.abs(A @ w_candidate_rounded - np.ones(m)))
                        if res < 1e-9:
                            W_sol[k * n + i, :] = w_candidate_rounded
                            found = True
                            break # Fixed: Break only if found
                    
                    if not found:
                        return None
                else:
                    # Full rank - unique solution, check if it's half-integer
                    w_exact = np.round(w_ls * 2) / 2
                    res = np.max(np.abs(A @ w_exact - np.ones(m)))
                    if res < 1e-9:
                        W_sol[k * n + i, :] = w_exact
                    else:
                        return None
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :] * V[j * p + k, :], W_sol[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return None
    
    return W_sol

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Check Rank 1 (All-Ones) first
    # U, V, W all ones
    U_1 = [[1] * 1 for _ in range(n * m)]
    V_1 = [[1] * 1 for _ in range(m * p)]
    W_1 = [[1] * 1 for _ in range(p * n)]
    
    if _verify(U_1, V_1, W_1, n, m, p):
        return {"U": U_1, "V": V_1, "W": W_1}

    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound
    # If it's All-Ones, LB is 1. If it's MM, LB is higher.
    # We already tried 1. So start from 2.
    lower_bound = 2
    
    np.random.seed(42)
    
    # Entry sets to try
    entry_sets = [
        np.array([-1, -0.5, 0, 0.5, 1]),
        np.array([-2, -1, -0.5, 0, 0.5, 1, 2]),
        np.array([0, 1]),
        np.array([-1, 0, 1]),
    ]
    
    # Try ranks from lower_bound up to best_rank - 1
    ranks_to_try = list(range(lower_bound, best_rank))
    # Prioritize lower ranks first now that 1 is checked
    # If 1 failed, it's likely MM, so 2, 3... might also fail, but we should try.
    # However, MM rank is usually much higher.
    # Let's try a few low ranks, then jump to the middle.
    
    # Sort by distance from a "likely" rank?
    # For MM 3x5x7, rank is likely > 35.
    # Let's just try in increasing order, but limit attempts for very low ranks.
    
    for R in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        if R >= best_rank:
            continue
        
        # Number of attempts
        max_attempts = 10 if R < 20 else (50 if R < 70 else 20)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Choose entry set
            es = entry_sets[attempt % len(entry_sets)]
            
            # Generate random U and V
            U_rand = np.random.choice(es, size=(n * m, R))
            V_rand = np.random.choice(es, size=(m * p, R))
            
            # Try to solve for W
            W_sol = _try_solve_for_W(U_rand, V_rand, n, m, p, R)
            
            if W_sol is not None:
                # Verify
                if _verify(U_rand.tolist(), V_rand.tolist(), W_sol.tolist(), n, m, p):
                    best_rank = R
                    best_solution = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_sol.tolist()
                    }
                    break  # Move to next rank
    
    return best_solution
