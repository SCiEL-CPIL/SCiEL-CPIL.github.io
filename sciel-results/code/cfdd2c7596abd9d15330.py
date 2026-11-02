"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit

    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    best_rank = rank_schoolbook

    # Precompute support sets for each dimension of the tensor
    # For index (i, j, k) in tensor T:
    # U index is i*m + j
    # V index is j*p + k
    # W index is k*n + i
    
    # We want to find a set of columns (u_idx, v_idx, w_idx) such that
    # for every (i, j, k) in {0..n-1} x {0..m-1} x {0..p-1},
    # there exists a column c such that:
    # U[i*m+j, c] != 0
    # V[j*p+k, c] != 0
    # W[k*n+i, c] != 0
    
    # This is a hitting set problem.
    # Elements: all (i, j, k)
    # Sets: for each column c, the set of (i, j, k) covered by non-zeros in U[:, c], V[:, c], W[:, c].
    # A column c covers (i, j, k) if U[i*m+j, c] != 0 AND V[j*p+k, c] != 0 AND W[k*n+i, c] != 0.
    
    # To make this tractable, we can restrict the search space for U, V, W entries.
    # Given the "Strassen" hint, maybe we can look for structured solutions?
    # Or just use the greedy hitting set approach with random restarts on the *values*.
    
    # Let's try a different approach:
    # 1. Generate random sparse U, V, W matrices with entries from {-1, 0, 1}.
    # 2. Check if the tensor is covered.
    # 3. If not, use local search to flip bits to cover more.
    
    # Actually, the power method approach in the current code tries to find *values* that sum to 1.
    # The hitting set approach finds *supports* that cover the tensor.
    # Once we have supports, we need to find values.
    # If supports form a "perfect" cover (each element covered exactly once), values are 1.
    # If not, we need to solve a linear system or use least squares.
    
    # Let's stick to the power method but improve the initialization and search.
    # The key insight from "Strassen recursion" might be that we should try to find
    # decompositions that are not just random, but have some structure.
    # However, for 2x5x6, the structure is not obvious.
    
    # Let's try to improve the local search by using a better initialization.
    # Instead of random integers, initialize with the schoolbook solution and then
    # try to merge columns?
    
    # Another idea: Use the fact that the tensor is sparse.
    # T has 60 non-zeros.
    # We want to approximate T with sum of r rank-1 tensors.
    
    # Let's try to search for ranks from 55 down to 40.
    # For each rank, use random restarts with the power method.
    
    # Improvement: 
    # 1. Use a time budget per rank.
    # 2. If a rank is not found quickly, move to the next lower rank? 
    #    No, lower rank is harder. 
    #    So we should spend more time on higher ranks first?
    #    Actually, if we find rank 55, we should try 54.
    #    If we fail to find 55, we should try 54.
    #    So the loop structure is correct.
    
    # Let's optimize the power method part.
    # The current code computes `current_sum` and `residual` in every iteration.
    # This is O(r * n*m*m*p*p*n) which is expensive.
    # We can maintain the sum incrementally.
    
    # Try structured initialization first
    def try_structured_init(r):
        """Try to find a solution with structured (non-random) initialization."""
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Initialize with schoolbook-like structure but compressed
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
        
        # Fill remaining columns with small random values
        for c in range(col, r):
            U[:, c] = np.random.randn(n*m) * 0.1
            V[:, c] = np.random.randn(m*p) * 0.1
            W[:, c] = np.random.randn(p*n) * 0.1
        
        return U, V, W
    
    for r in range(59, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple restarts for this rank
        # Allocate more time for higher ranks (easier to find)
        if r >= 55:
            num_restarts = 50
            max_iters = 80
        elif r >= 50:
            num_restarts = 80
            max_iters = 100
        else:
            num_restarts = 120
            max_iters = 120
            
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Choose initialization strategy
            if restart < 5:
                # Try structured initialization
                U, V, W = try_structured_init(r)
            elif restart < 15:
                # Sparse random initialization
                U = np.random.randint(-1, 2, (n*m, r)).astype(float)
                V = np.random.randint(-1, 2, (m*p, r)).astype(float)
                W = np.random.randint(-1, 2, (p*n, r)).astype(float)
            else:
                # Dense random initialization with smaller magnitude
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            
            # Precompute initial sum
            current_sum = np.einsum('ic,jc,kc->ijk', U, V, W)
            
            converged = False
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                    
                residual = T - current_sum
                residual_norm = np.linalg.norm(residual)
                
                # Check convergence
                if residual_norm < 1e-9:
                    converged = True
                    break
                
                # Multiple update strategies
                if it % 3 == 0:
                    # Single random column update
                    c = np.random.randint(r)
                elif it % 3 == 1:
                    # Update column with largest contribution to residual
                    col_contributions = []
                    for cc in range(min(r, 20)):  # Sample some columns
                        col_sum = np.einsum('i,j,k->ijk', U[:, cc], V[:, cc], W[:, cc])
                        contrib = np.sum(residual * col_sum)
                        col_contributions.append((abs(contrib), cc))
                    col_contributions.sort(reverse=True)
                    c = col_contributions[0][1]
                else:
                    # Random column
                    c = np.random.randint(r)
                
                # Compute residual excluding column c
                col_sum = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                target_for_c = residual + col_sum
                
                # Power method to find best rank-1 approx to target_for_c
                u = U[:, c].copy()
                v = V[:, c].copy()
                w = W[:, c].copy()
                
                # If vectors are zero, reinitialize
                if np.linalg.norm(u) < 1e-6: u = np.random.randn(n*m)
                if np.linalg.norm(v) < 1e-6: v = np.random.randn(m*p)
                if np.linalg.norm(w) < 1e-6: w = np.random.randn(p*n)
                
                # More power iterations for better convergence
                for _ in range(15):
                    # u_new = target_for_c contracted with v, w
                    u_new = np.einsum('ijk,j,k->i', target_for_c, v, w)
                    v_new = np.einsum('ijk,i,k->j', target_for_c, u, w)
                    w_new = np.einsum('ijk,i,j->k', target_for_c, u, v)
                    
                    norm_u = np.linalg.norm(u_new)
                    norm_v = np.linalg.norm(v_new)
                    norm_w = np.linalg.norm(w_new)
                    
                    if norm_u < 1e-10 or norm_v < 1e-10 or norm_w < 1e-10:
                        break
                    
                    u_new /= norm_u
                    v_new /= norm_v
                    w_new /= norm_w
                    
                    u, v, w = u_new, v_new, w_new
                
                # Scale to match magnitude
                uo = np.einsum('i,j,k->ijk', u, v, w)
                denom = np.sum(uo * uo)
                if denom < 1e-10:
                    continue
                s = np.sum(target_for_c * uo) / denom
                
                # Update column c
                U[:, c] = s * u
                V[:, c] = s * v
                W[:, c] = s * w
                
                # Update current_sum incrementally
                new_col = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                current_sum = current_sum - col_sum + new_col
                
            if converged:
                # We have a real-valued solution.
                # Try to round to half-integers with multiple strategies
                found = False
                
                # Strategy 1: Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(check_sum, T, atol=1e-9):
                    best_U = U_rounded.tolist()
                    best_V = V_rounded.tolist()
                    best_W = W_rounded.tolist()
                    best_rank = r
                    found = True
                else:
                    # Strategy 2: Try rounding with small perturbations
                    for _ in range(10):
                        U_pert = U + np.random.randn(n*m, r) * 0.01
                        V_pert = V + np.random.randn(m*p, r) * 0.01
                        W_pert = W + np.random.randn(p*n, r) * 0.01
                        
                        U_rounded = np.round(U_pert * 2) / 2
                        V_rounded = np.round(V_pert * 2) / 2
                        W_rounded = np.round(W_pert * 2) / 2
                        
                        check_sum = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
                        if np.allclose(check_sum, T, atol=1e-9):
                            best_U = U_rounded.tolist()
                            best_V = V_rounded.tolist()
                            best_W = W_rounded.tolist()
                            best_rank = r
                            found = True
                            break
                
                if found:
                    break
        # Continue to next lower rank

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
