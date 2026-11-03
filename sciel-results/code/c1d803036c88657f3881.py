"""Combined approach: deterministic schoolbook fallback + randomized Newton-Raphson
polynomial system solver for rank-minimal 4x4 x 4x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Schoolbook fallback: rank 80
    best_rank = n * m * p  # 80
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Build target tensor
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k), else 0
    # a = i*m + j, b = j*p + k, c = k*n + i
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # The decomposition requires:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    # This is a system of polynomial equations in the variables U, V, W.
    
    # Randomized Newton-Raphson approach with improved convergence:
    # - Better initialization strategies
    # - Adaptive step size with backtracking
    # - Multiple restarts with different seeds
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Try ranks from lower values upward
    # For 4x4x4, Laderman gives rank 49. For 4x4x5, try from 60 upward
    ranks_to_try = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78]
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue
        
        # Try multiple random initializations
        max_trials = 12 if r <= 68 else 6
        
        for trial in range(max_trials):
            if time.time() > deadline - 3:
                break
            
            # Multiple initialization strategies
            strategy = trial % 3
            if strategy == 0:
                # Standard Gaussian
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            elif strategy == 1:
                # Sparse initialization - only some entries non-zero
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for rr in range(r):
                    # Randomly select a few indices for each factor
                    idx_u = np.random.choice(n*m, size=3, replace=False)
                    idx_v = np.random.choice(m*p, size=3, replace=False)
                    idx_w = np.random.choice(p*n, size=3, replace=False)
                    U[idx_u, rr] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    V[idx_v, rr] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
                    W[idx_w, rr] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
            else:
                # Block structure initialization
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                block_size = max(1, r // 4)
                for b in range(4):
                    start = b * block_size
                    end = min(start + block_size, r)
                    if start < end:
                        # Assign blocks with some structure
                        for rr in range(start, end):
                            U[np.random.choice(n*m, size=4, replace=False), rr] = 1.0
                            V[np.random.choice(m*p, size=4, replace=False), rr] = 1.0
                            W[np.random.choice(p*n, size=4, replace=False), rr] = 1.0
            
            # Newton-Raphson iteration with improved line search
            lr_base = 0.1
            max_iter = 500
            converged = False
            min_res = np.inf
            stagnation_count = 0
            
            for step in range(max_iter):
                if time.time() > deadline - 2:
                    break
                
                # Compute current tensor estimate
                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-10:
                    converged = True
                    break
                
                if max_res < min_res:
                    min_res = max_res
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                
                if stagnation_count > 50:
                    # Try a larger perturbation to escape local minimum
                    U += np.random.randn(n*m, r) * 0.1
                    V += np.random.randn(m*p, r) * 0.1
                    W += np.random.randn(p*n, r) * 0.1
                    stagnation_count = 0
                    continue
                
                # Compute gradients
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adaptive learning rate
                lr = lr_base / (1.0 + 0.05 * step)
                
                # Backtracking line search
                for backtrack in range(10):
                    U_new = U - lr * dU
                    V_new = V - lr * dV
                    W_new = W - lr * dW
                    
                    est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    res_new = np.max(np.abs(est_new - target))
                    
                    if res_new < max_res:
                        U, V, W = U_new, V_new, W_new
                        break
                    else:
                        lr *= 0.5
                
                # Check for convergence periodically
                if step % 25 == 24:
                    est_check = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(est_check - target)) < 1e-9:
                        converged = True
                        break
            
            # Final verification with high precision
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(est_final - target)) < 1e-8:
                # Found a valid decomposition!
                # Try to clean up to integers/half-integers with multiple rounding strategies
                best_clean = None
                best_clean_err = np.inf
                
                for factor in [1, 2, 4, 8]:
                    U_clean = np.round(U * factor) / factor
                    V_clean = np.round(V * factor) / factor
                    W_clean = np.round(W * factor) / factor
                    
                    est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                    err = np.max(np.abs(est_clean - target))
                    
                    if err < 1e-7 and err < best_clean_err:
                        best_clean = (U_clean, V_clean, W_clean)
                        best_clean_err = err
                
                if best_clean is not None:
                    best_rank = r
                    U_c, V_c, W_c = best_clean
                    best_result = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                    break
                else:
                    # Use the float version if it's valid
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank < n * m * p:
            # Found improvement, try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
