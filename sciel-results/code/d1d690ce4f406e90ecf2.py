import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Check if the tensor decomposition is valid.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Target tensor T[i, j, k] = 1 if i == k else 0.
    Note: W index is k*n + i.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False

    # Efficient check using vectorized operations per (i, k) pair
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can compute the full tensor T and compare.
    # T is shape (n, m, p)
    # Let's use loops for i, k since n, p are small (3, 5).
    # Inner loop over j and r can be vectorized.
    
    for i in range(n):
        for k in range(p):
            # U_i: shape (m, rank) -> U[i*m : (i+1)*m, :]
            U_i = U[i*m : (i+1)*m, :]
            # W_k: shape (rank) -> W[k*n + i, :]
            W_k = W[k*n + i, :]
            
            # We need T[i, j, k] for all j in 0..m-1
            # T[i, j, k] = sum_r U_i[j, r] * V[j*p+k, r] * W_k[r]
            
            # V_jk: shape (m, rank)
            # V[j, p*k : p*(k+1), :] -> reshape to (m, rank)
            V_slice = V[:, k*p : (k+1)*p].T # Shape (p, m*?) No.
            # V is (m*p, rank). Index j*p + k.
            # So for fixed k, we want V[j*p + k, :] for all j.
            # This is not a contiguous slice in V.
            # V indices for fixed k: k, p+k, 2p+k, ...
            # So V_slice[j] = V[j*p + k, :]
            
            V_jk = np.zeros((m, rank))
            for j in range(m):
                V_jk[j] = V[j*p + k, :]
            
            # Now T_i_k[j] = sum_r U_i[j, r] * V_jk[j, r] * W_k[r]
            # This is element-wise product of U_i and V_jk, then dot with W_k.
            # U_i * V_jk is (m, rank).
            prod = U_i * V_jk
            T_i_k = prod @ W_k # Shape (m,)
            
            target = 1 if i == k else 0
            if not np.allclose(T_i_k, target, atol=tol, rtol=0):
                return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    # 1. Initialize with Schoolbook (Rank 45)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_seed.copy(), V_seed.copy(), W_seed.copy()
    best_rank = rank_seed
    
    # 2. Greedy Removal Strategy
    # Try to remove terms one by one.
    # To speed up, we can try removing terms that are likely redundant.
    # In the schoolbook, every term is unique. So simple removal won't work on seed.
    # We need a smarter initialization or a perturbation.
    
    # Alternative: Start with a known low-rank structure if possible?
    # For 3x3x5, rank is likely around 30-40?
    # Let's try to find a decomposition of rank R by random search + refinement?
    # Too slow for 120s.
    
    # Let's try a specific structural decomposition.
    # C = A B.
    # A is 3x3, B is 3x5.
    # We can view this as 5 column multiplications A * b_j.
    # Each is a 3x1 vector resulting from 3x3 matrix.
    # Rank of 3x3x1 is 9 (schoolbook). Total 45.
    # Can we do better?
    
    # Let's try to use the fact that we can combine some columns.
    # This is essentially finding a lower rank decomposition.
    
    # Since numerical optimization is hard, let's try a randomized "greedy" approach on a perturbed seed.
    # Or better, try to construct a rank 40 decomposition manually?
    
    # Actually, let's try a simple local search:
    # Start with seed.
    # Randomly perturb a few entries.
    # Try to remove a term? No, perturbation breaks exactness.
    
    # Let's stick to the most robust method:
    # 1. Check if we can find a valid decomposition of rank < 45 using a small grid search on a subset of parameters?
    # No.
    
    # Let's try the "Inspiration 2" approach: Greedy Removal on the seed.
    # It won't remove anything from the pure seed because all terms are linearly independent in the tensor space?
    # Actually, for matrix multiplication tensor, the schoolbook terms are linearly independent.
    # So greedy removal on seed yields 0 improvements.
    
    # We need a different starting point.
    # Let's try to construct a decomposition using block diagonals or similar?
    
    # Idea: 
    # The tensor is T_{ijk} = delta_{ik}.
    # This is the identity tensor for the first and third indices, independent of j.
    # Wait, T_{ijk} = 1 if i=k.
    # This means the tensor is a block diagonal matrix if we view it as a matrix of size (n*p) x (m*n*p)? No.
    # Let's reshape T to a matrix M of size (n*m) x (p*n)? No.
    # Standard unfolding:
    # T is a 3-tensor (n, m, p).
    # Matrix unfolding (n) x (m*p): M_{i, jk} = T_{ijk}.
    # M_{i, jk} = 1 if i=k else 0.
    # This matrix has rank n (since for each i, the row is non-zero only at columns where k=i).
    # Specifically, row i has 1s at columns (j, i) for all j.
    # So rank is n=3.
    # This implies the tensor rank is at least 3? No, tensor rank is different.
    # But it suggests a structure.
    
    # Let's try to exploit the structure.
    # T_{ijk} = delta_{ik}.
    # We want sum_r U_{ij}^r V_{jk}^r W_{ki}^r = delta_{ik}.
    
    # Consider a decomposition where U, V, W are sparse.
    
    # Let's try a rank 3n = 9 decomposition?
    # If we set r from 0 to 8.
    # Let r = i*3 + j?
    # This doesn't seem to work directly for the W part.
    
    # Let's try a rank 45 seed, and then use a randomized perturbation + local search to find a lower rank?
    # Given the time limit, and the difficulty, I will implement a robust fallback that attempts a few random restarts of a simple local search (gradient descent on the residual) for lower ranks, but with a very tight budget.
    
    # However, the prompt asks to combine strategies.
    # Inspiration 2 has greedy removal.
    # Inspiration 3 has efficient checking.
    # Current program has the seed.
    
    # Let's implement a "Randomized Greedy Removal" where we first add some random "noise" terms to the seed, then try to remove terms? No, that increases rank.
    
    # What if we start with a rank 45 seed, and try to merge terms?
    
    # Actually, there is a known result: rank of M_{3,3,3} is 23.
    # M_{3,3,5} can be seen as M_{3,3,3} + M_{3,3,2}.
    # Rank(M_{3,3,2}) <= 18.
    # Total 41.
    # Can we do better?
    
    # Let's try to hardcode a Rank 41 decomposition if possible?
    # I don't have the explicit 23-rank matrices.
    
    # Let's try a simple optimization loop for Rank 40.
    # Variables: 9*40 + 15*40 + 15*40 = 1560.
    # Objective: sum (val - target)^2.
    # Use Nelder-Mead? Too slow for 1560 dims.
    # Use L-BFGS-B with gradients?
    # Gradient of E wrt U_{ij}^r is 2 * (val_{ijk} - target_{ijk}) * V_{jk}^r * W_{ki}^r summed over k?
    # Yes, gradients are easy to compute.
    
    # Let's implement a gradient descent for Rank 40 and 35.
    
    def compute_error_and_grad(params, n, m, p, rank):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U = params[:size_u].reshape((n * m, rank))
        V = params[size_u:size_u + size_v].reshape((m * p, rank))
        W = params[size_u + size_v:].reshape((p * n, rank))
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        error = 0.0
        
        # Precompute T and residuals
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can accumulate gradients in a loop over i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    
                    target = 1 if i == k else 0
                    diff = val - target
                    error += diff * diff
                    
                    # Gradient contributions
                    # dE/du = 2 * diff * V * W
                    for r in range(rank):
                        grad_U[i * m + j, r] += 2 * diff * V[j * p + k, r] * W[k * n + i, r]
                        grad_V[j * p + k, r] += 2 * diff * U[i * m + j, r] * W[k * n + i, r]
                        grad_W[k * n + i, r] += 2 * diff * U[i * m + j, r] * V[j * p + k, r]
        
        return error, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    from scipy.optimize import minimize
    
    # Try to find a solution for Rank 40
    target_rank = 40
    if time.time() < deadline - 20:
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        total_size = size_u + size_v + size_w
        
        # Initialize with seed terms + random noise?
        # Better: Initialize with the first 40 terms of the seed?
        # That leaves 5 terms missing. The error will be high.
        # But maybe gradient descent can fix it?
        
        x0 = np.zeros(total_size)
        # Copy first 40 terms of seed
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    x0[col * rank + i * m + j] = 1 # Wait, params are flattened U, V, W.
                    # U is size_u.
                    # U[i*m+j, col] = 1
                    x0[i * m + j + col * n * m] = 1
                    # V[j*p+k, col] = 1
                    x0[size_u + j * p + k + col * m * p] = 1
                    # W[k*n+i, col] = 1
                    x0[size_u + size_v + k * n + i + col * p * n] = 1
                    col += 1
        
        # Add small random noise to help escape local minima?
        # Or just run it.
        
        try:
            res = minimize(compute_error_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6})
            if res.success and res.fun < 1e-6:
                # Verify
                U_new = res.x[:size_u].reshape((n * m, target_rank))
                V_new = res.x[size_u:size_u + size_v].reshape((m * p, target_rank))
                W_new = res.x[size_u + size_v:].reshape((p * n, target_rank))
                if check_decomposition(U_new, V_new, W_new, n, m, p):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
        except Exception as e:
            pass
            
    # If we found rank 40, try 35?
    if best_rank == 40 and time.time() < deadline - 10:
        target_rank = 35
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        total_size = size_u + size_v + size_w
        
        x0 = np.random.uniform(-0.1, 0.1, total_size)
        
        try:
            res = minimize(compute_error_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-6})
            if res.success and res.fun < 1e-6:
                U_new = res.x[:size_u].reshape((n * m, target_rank))
                V_new = res.x[size_u:size_u + size_v].reshape((m * p, target_rank))
                W_new = res.x[size_u + size_v:].reshape((p * n, target_rank))
                if check_decomposition(U_new, V_new, W_new, n, m, p):
                    best_U, best_V, best_W = U_new, V_new, W_new
                    best_rank = target_rank
        except Exception as e:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
