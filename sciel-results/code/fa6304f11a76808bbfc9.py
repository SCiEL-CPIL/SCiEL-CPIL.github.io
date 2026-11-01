"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from numpy import allclose
import time
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # The matrix multiplication tensor T has entries T[i,j,k] where
    # i in [0,n), j in [0,m), k in [0,p)
    # A rank-r decomposition: sum_{c=0}^{r-1} u_c * v_c * w_c
    # where u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n
    # 
    # The tensor entry (i,j,k) should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Build the target tensor (flattened)
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook solution as fallback (rank = n*m*p = 42)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    def check_validity(U, V, W, tol=1e-8):
        """Check if the decomposition is valid."""
        r = U.shape[1]
        if r != V.shape[1] or r != W.shape[1]:
            return False
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return False
        # Compute the reconstructed tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_list(U, V, W):
        """Convert numpy arrays to lists for the return format."""
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Strategy: Try local search with decreasing rank
    # Start from a feasible point and try to reduce rank
    
    best_result = to_list(U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try rank values from low to high
    # Known: R(M_{2,3,7}) is between 25 and 42
    # Try ranks: 25, 26, 27, ..., up to 42
    target_ranks = list(range(25, rank_school + 1))
    
    def try_rank(r, max_iterations=50000, restarts=5):
        """Try to find a valid decomposition of rank r."""
        nonlocal best_result, best_rank
        
        for restart in range(restarts):
            if time.time() - start_time > deadline - 5:
                return False
            
            # Initialize: use first r schoolbook columns + random perturbation
            if r <= rank_school:
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                
                # Use some schoolbook terms as basis
                num_school = min(r, 10)  # use up to 10 schoolbook terms
                for c in range(num_school):
                    i = c // (m*p)
                    rem = c % (m*p)
                    j = rem // p
                    k = rem % p
                    U[i*m + j, c] = 1.0
                    V[j*p + k, c] = 1.0
                    W[k*n + i, c] = 1.0
                
                # Fill remaining columns with random values
                for c in range(num_school, r):
                    U[:, c] = np.random.randn(n*m) * 0.5
                    V[:, c] = np.random.randn(m*p) * 0.5
                    W[:, c] = np.random.randn(p*n) * 0.5
            else:
                return False
            
            # Local search: gradient descent on the objective
            # Objective: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # Plus regularization to keep values reasonable
            
            lr = 0.01
            lr_decay = 0.999
            best_obj = float('inf')
            converged = False
            
            for it in range(max_iterations):
                if time.time() - start_time > deadline - 5:
                    return False
                if it % 10000 == 0 and time.time() - start_time > deadline - 10:
                    break
                
                # Compute objective and gradients
                # For efficiency, compute the full tensor
                obj = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m + j
                            idx_v = j*p + k
                            idx_w = k*n + i
                            val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                            diff = val - 1.0
                            obj += diff * diff
                            
                            # Gradients
                            # d/dU[idx_u,c] = 2*diff * V[idx_v,c] * W[idx_w,c]
                            # d/dV[idx_v,c] = 2*diff * U[idx_u,c] * W[idx_w,c]
                            # d/dW[idx_w,c] = 2*diff * U[idx_u,c] * V[idx_v,c]
                            grad_U[idx_u, :] += 2.0 * diff * (V[idx_v, :] * W[idx_w, :])
                            grad_V[idx_v, :] += 2.0 * diff * (U[idx_u, :] * W[idx_w, :])
                            grad_W[idx_w, :] += 2.0 * diff * (U[idx_u, :] * V[idx_v, :])
                
                if obj < 1e-12:
                    converged = True
                    break
                
                # Update with step size
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= lr_decay
                
                # If objective increased, reduce lr
                if obj > best_obj * 1.1:
                    lr *= 0.5
                else:
                    best_obj = min(best_obj, obj)
                
                if converged:
                    break
            
            # Check if we found a valid solution
            if check_validity(U, V, W, tol=1e-8):
                if r < best_rank:
                    best_rank = r
                    best_result = to_list(U, V, W)
                    return True
                else:
                    return True
        
        return False
    
    # Try ranks in increasing order
    for r in target_ranks:
        if time.time() - start_time > deadline - 10:
            break
        
        # For higher ranks, we have more confidence
        # For lower ranks, be more careful
        if r <= 30:
            restarts = 3
            max_iter = 30000
        elif r <= 38:
            restarts = 2
            max_iter = 20000
        else:
            restarts = 1
            max_iter = 10000
        
        success = try_rank(r, max_iterations=max_iter, restarts=restarts)
        if success:
            # Found a valid solution at this rank
            # We could try to go lower, but let's also verify
            if check_validity(np.array(best_result["U"]), np.array(best_result["V"]), np.array(best_result["W"])):
                break
        else:
            # Couldn't find at this rank, try higher
            continue
    
    # Final validation of the best result
    U_final = np.array(best_result["U"])
    V_final = np.array(best_result["V"])
    W_final = np.array(best_result["W"])
    
    if not check_validity(U_final, V_final, W_final, tol=1e-8):
        # Fall back to schoolbook
        best_result = to_list(U_school, V_school, W_school)
        best_rank = rank_school
    
    return best_result
# EVOLVE-BLOCK-END
