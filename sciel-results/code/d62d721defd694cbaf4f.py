"""Combined approach: deterministic structured rank reduction + polynomial system solver
for rank-minimal 4x4 x 4x5 matrix multiplication tensor."""
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
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    # Strategy 1: Try structured decompositions with known algebraic patterns
    # For matrix multiplication, we can try to merge some schoolbook terms
    # by exploiting linearity in certain indices.
    
    # Attempt rank 75: merge some terms using the fact that for fixed (i,j),
    # the k-index appears linearly in V and W.
    # Specifically, for each (i,j) pair, we have p=5 terms. We can try to
    # represent some of these with fewer rank-1 tensors by sharing structure.
    
    # Let's try a more systematic approach: use the fact that the tensor
    # T[i,j,k] has a specific structure and try to find low-rank approximations
    # that are exact.
    
    # Strategy: Try ranks from 70 to 79, using a combination of:
    # 1. Start with schoolbook and try to merge pairs of terms
    # 2. Use random initialization + Newton's method for polynomial system
    
    random.seed(42)
    np.random.seed(42)
    
    ranks_to_try = [70, 72, 74, 75, 76, 77, 78, 79]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        
        # Multiple random initializations with Newton-Raphson
        max_trials = 12 if r <= 75 else 6
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a mix of structured and random components
            # Start close to schoolbook but with perturbations
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Add some structured components to help convergence
            # For the first min(r, n*m*p) columns, initialize with schoolbook-like patterns
            for col in range(min(r, n*m*p)):
                # Distribute the schoolbook terms across the rank
                i = (col // (m*p)) % n
                j = (col // p) % m
                k = col % p
                U[i*m + j, col] += 1.0
                V[j*p + k, col] += 1.0
                W[k*n + i, col] += 1.0
            
            # Newton-Raphson iteration with adaptive step size
            lr_base = 0.1
            max_iter = 500
            converged = False
            prev_res = np.inf
            
            for step in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Compute current tensor estimate and residual
                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-12:
                    converged = True
                    break
                
                # Check for stagnation
                if max_res > prev_res * 0.999 and step > 50:
                    # Try random perturbation to escape local minima
                    pert = 0.05
                    U += np.random.randn(n*m, r) * pert
                    V += np.random.randn(m*p, r) * pert
                    W += np.random.randn(p*n, r) * pert
                    prev_res = max_res
                    continue
                
                prev_res = max_res
                
                # Compute gradients
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Adaptive learning rate
                lr = lr_base / (1.0 + 0.05 * step)
                
                # Line search with backtracking
                accepted = False
                for ls in range(10):
                    lr_try = lr * (0.5 ** ls)
                    
                    U_new = U - lr_try * dU
                    V_new = V - lr_try * dV
                    W_new = W - lr_try * dW
                    
                    est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    res_new = np.max(np.abs(est_new - target))
                    
                    if res_new < max_res:
                        U, V, W = U_new, V_new, W_new
                        max_res = res_new
                        accepted = True
                        break
                    else:
                        lr = lr_try
                
                if not accepted:
                    # If no step was accepted, add small random perturbation
                    U += np.random.randn(n*m, r) * 0.01
                    V += np.random.randn(m*p, r) * 0.01
                    W += np.random.randn(p*n, r) * 0.01
            
            # Final verification
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(est_final - target)) < 1e-9:
                # Found a valid decomposition!
                # Try to clean up to integers/half-integers
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                if np.max(np.abs(est_clean - target)) < 1e-7:
                    best_rank = r
                    best_result = {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
                    break
                else:
                    # Use the float version if it's valid
                    best_rank = r
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    # Strategy 2: Try to further reduce rank using structured merging
    # For the best rank found, try to see if we can merge more terms
    if best_rank < n * m * p and time.time() < deadline - 15:
        # Try to reduce by 1-2 more ranks using local optimization
        for target_rank in range(max(1, best_rank - 2), best_rank):
            if time.time() > deadline - 10:
                break
            
            # Use the current best solution as initialization
            U_init = np.array(best_result["U"])
            V_init = np.array(best_result["V"])
            W_init = np.array(best_result["W"])
            
            # Pad or truncate to target_rank
            if U_init.shape[1] > target_rank:
                U_init = U_init[:, :target_rank]
                V_init = V_init[:, :target_rank]
                W_init = W_init[:, :target_rank]
            elif U_init.shape[1] < target_rank:
                pad_u = np.random.randn(n*m, target_rank - U_init.shape[1]) * 0.1
                pad_v = np.random.randn(m*p, target_rank - V_init.shape[1]) * 0.1
                pad_w = np.random.randn(p*n, target_rank - W_init.shape[1]) * 0.1
                U_init = np.hstack([U_init, pad_u])
                V_init = np.hstack([V_init, pad_v])
                W_init = np.hstack([W_init, pad_w])
            
            # Optimize from this initialization
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            lr_base = 0.05
            max_iter = 300
            converged = False
            prev_res = np.inf
            
            for step in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                est = np.einsum('ar,br,cr->abc', U, V, W)
                residual = est - target
                max_res = np.max(np.abs(residual))
                
                if max_res < 1e-12:
                    converged = True
                    break
                
                if max_res > prev_res * 0.999 and step > 30:
                    U += np.random.randn(n*m, target_rank) * 0.02
                    V += np.random.randn(m*p, target_rank) * 0.02
                    W += np.random.randn(p*n, target_rank) * 0.02
                    prev_res = max_res
                    continue
                
                prev_res = max_res
                
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                lr = lr_base / (1.0 + 0.1 * step)
                
                accepted = False
                for ls in range(8):
                    lr_try = lr * (0.5 ** ls)
                    
                    U_new = U - lr_try * dU
                    V_new = V - lr_try * dV
                    W_new = W - lr_try * dW
                    
                    est_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    res_new = np.max(np.abs(est_new - target))
                    
                    if res_new < max_res:
                        U, V, W = U_new, V_new, W_new
                        max_res = res_new
                        accepted = True
                        break
                    else:
                        lr = lr_try
                
                if not accepted:
                    U += np.random.randn(n*m, target_rank) * 0.005
                    V += np.random.randn(m*p, target_rank) * 0.005
                    W += np.random.randn(p*n, target_rank) * 0.005
            
            est_final = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(est_final - target)) < 1e-9:
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                est_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                if np.max(np.abs(est_clean - target)) < 1e-7:
                    best_rank = target_rank
                    best_result = {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
                else:
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
