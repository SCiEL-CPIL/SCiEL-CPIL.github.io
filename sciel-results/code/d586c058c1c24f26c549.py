import time
import numpy as np

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 105
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def verify_exact(U, V, W, n, m, p):
        """
        Exact verification using numpy einsum for speed.
        T[a,b,c] = sum_r U[b,r] * V[c,r] * W[a,r]
        where a=k*n+i, b=i*m+j, c=j*p+k
        """
        T = np.einsum('br,cr,ar->abc', U, V, W)
        
        # Check diagonal positions (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = k * n + i
                    b = i * m + j
                    c = j * p + k
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        return False
        
        # Check all other positions (should be 0)
        for a in range(p * n):
            k_a = a // n
            i_a = a % n
            for b in range(n * m):
                ii_b = b // m
                j_b = b % m
                for c in range(m * p):
                    jj_c = c // p
                    kk_c = c % p
                    if not (i_a == ii_b and j_b == jj_c and k_a == kk_c):
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True
    
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    def _try_solve_for_W(U_mat, V_mat, n, m, p, R):
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
                    A[j, :] = U_mat[i * m + j, :] * V_mat[j * p + k, :]
                
                # Solve A @ w = ones(m)
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
                    if rank_A < m:
                        if rank_A == 0:
                            return None
                        U_svd, S, Vt = np.linalg.svd(A, full_matrices=True)
                        null_space = Vt[rank_A:, :].T
                        
                        found = False
                        for attempt in range(50):
                            t = np.random.randint(-4, 5, size=(R - rank_A,)) * 0.5
                            w_candidate = w_ls + null_space @ t
                            w_candidate_rounded = np.round(w_candidate * 2) / 2
                            res = np.max(np.abs(A @ w_candidate_rounded - np.ones(m)))
                            if res < 1e-9:
                                W_sol[k * n + i, :] = w_candidate_rounded
                                found = True
                                break
                        
                        if not found:
                            return None
                    else:
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
                    val = np.dot(U_mat[i * m + j, :] * V_mat[j * p + k, :], W_sol[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return None
        
        return W_sol
    
    np.random.seed(42)
    
    # Entry sets to try - focused on small values for half-integer solutions
    entry_sets = [
        np.array([-1, -0.5, 0, 0.5, 1]),
        np.array([-2, -1, -0.5, 0, 0.5, 1, 2]),
        np.array([0, 1]),
        np.array([-1, 0, 1]),
        np.array([-1, 1]),
    ]
    
    # Try ranks from lower_bound up to best_rank - 1
    # Focus on ranks that are more likely to work based on known bounds
    ranks_to_try = list(range(lower_bound, best_rank))
    # Prioritize middle ranks first, then extremes
    mid_rank = (lower_bound + best_rank) // 2
    ranks_to_try.sort(key=lambda R: (abs(R - mid_rank), R))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        if R >= best_rank:
            continue
        
        # Number of attempts - more for lower ranks
        max_attempts = 30 if R < 60 else 15
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Choose entry set
            es = entry_sets[attempt % len(entry_sets)]
            
            # Generate random U and V
            U_rand = np.random.choice(es, size=(n * m, R))
            V_rand = np.random.choice(es, size=(m * p, R))
            
            # Try to solve for W
            W_sol = _try_solve_for_W(U_rand, V_rand, n, m, p, R)
            
            if W_sol is not None:
                # Verify with full tensor check
                if verify_exact(U_rand, V_rand, W_sol, n, m, p):
                    best_rank = R
                    best_result = {
                        'U': U_rand.tolist(),
                        'V': V_rand.tolist(),
                        'W': W_sol.tolist()
                    }
                    break  # Move to next rank
    
    return best_result
