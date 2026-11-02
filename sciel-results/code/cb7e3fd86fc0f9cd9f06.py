"""Tensor decomposition search for 3x3x5 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def _compute_residual(U, V, W):
    """Compute the residual tensor norm for a given decomposition."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target: T[i,j,k] = 1 if k==k (always true for this structure), but actually
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard basis
    # Wait, let me reconsider. The tensor for (3x3)*(3x5) -> (3x5) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,5)
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                diff = s - 1.0
                residual += diff * diff
    return residual

def _try_rank(rank, time_budget, rng):
    """Try to find a decomposition of given rank using gradient descent."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    U = rng.standard_normal((nm, rank)) * 0.5
    V = rng.standard_normal((mp, rank)) * 0.5
    W = rng.standard_normal((pn, rank)) * 0.5
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    best_residual = _compute_residual(U, V, W)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start = time.time()
    while time.time() - start < time_budget:
        # Compute gradients numerically (finite differences)
        # This is expensive but for small dimensions it's feasible
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        eps = 1e-5
        # Sample a subset of elements to compute gradients
        num_samples = min(50, nm * rank // 10)
        if num_samples < 1:
            num_samples = 1
        
        # For efficiency, compute full gradients but stop early if too slow
        for _ in range(10):  # Limit gradient computations per step
            # Gradient for U
            idx = rng.integers(0, nm * rank)
            i, r = divmod(idx, rank)
            orig = U[i, r]
            U[i, r] = orig + eps
            res_plus = _compute_residual(U, V, W)
            U[i, r] = orig - eps
            res_minus = _compute_residual(U, V, W)
            U[i, r] = orig
            grad_U[i, r] = (res_plus - res_minus) / (2 * eps)
            
            # Gradient for V
            idx = rng.integers(0, mp * rank)
            j, r = divmod(idx, rank)
            orig = V[j, r]
            V[j, r] = orig + eps
            res_plus = _compute_residual(U, V, W)
            V[j, r] = orig - eps
            res_minus = _compute_residual(U, V, W)
            V[j, r] = orig
            grad_V[j, r] = (res_plus - res_minus) / (2 * eps)
            
            # Gradient for W
            idx = rng.integers(0, pn * rank)
            k, r = divmod(idx, rank)
            orig = W[k, r]
            W[k, r] = orig + eps
            res_plus = _compute_residual(U, V, W)
            W[k, r] = orig - eps
            res_minus = _compute_residual(U, V, W)
            W[k, r] = orig
            grad_W[k, r] = (res_plus - res_minus) / (2 * eps)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        current_residual = _compute_residual(U, V, W)
        if current_residual < best_residual:
            best_residual = current_residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if best_residual < 1e-6:
                break
        
        # Adaptive learning rate
        if current_residual > best_residual * 1.5:
            lr *= 0.5
        elif current_residual < best_residual * 0.9:
            lr *= 1.1
    
    if best_residual < 1e-4:
        return best_U, best_V, best_W, best_residual
    return None, None, None, best_residual

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known lower bounds: For 3x3x5, the minimum rank is at least 27 (by Strassen's bound)
    # Let's try ranks from 27 up to 44
    
    for target_rank in range(27, min(45, best_rank)):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        rng = np.random.default_rng(target_rank * 1000 + 42)
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            time_budget = min(15, deadline - time.time() - 5)
            if time_budget < 1:
                break
            
            U, V, W, residual = _try_rank(target_rank, time_budget, rng)
            
            if U is not None and residual < 1e-4:
                # Verify the decomposition
                n, m, p = DIMENSIONS
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            if abs(s - 1.0) > 1e-3:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to list of lists
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
        
        # If we found a lower rank, continue trying even lower
        if best_rank <= target_rank:
            continue
    
    return best_result
