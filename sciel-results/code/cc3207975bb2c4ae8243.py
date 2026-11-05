"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact using numpy einsum."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Compute the full tensor R[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j, b = j'*p + k', c = k''*n + i''
    # We want R to be 1 if (i,j,k) matches (i'',j'',k'') consistently, 0 otherwise.
    # Specifically, R[i*m+j, j*p+k, k*n+i] should be 1 for all i,j,k.
    # And R should be 0 elsewhere.
    
    # Efficiently construct the target tensor
    dim_U, dim_V, dim_W = n*m, m*p, p*n
    T_target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Compute actual tensor
    R = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Check exactness
    if not np.allclose(R, T_target, atol=tol):
        return False
    return True

def _try_rank_optimize(n, m, p, target_rank, max_iters=200, tol=1e-6):
    """
    Try to find a decomposition of given rank using gradient-based optimization
    with structured initialization.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Multiple restarts with different initializations
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize
        if restart == 0:
            # Structured initialization based on two (3,3,3) blocks
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            half_rank = target_rank // 2
            rem = target_rank % 2
            
            for block in range(2):
                start = block * half_rank + min(block, rem)
                end = start + half_rank + (1 if block < rem else 0)
                block_size = end - start
                if block_size <= 0: continue
                
                # Random init for this block
                U_block = np.random.randn(nm, block_size) * 0.5
                V_block = np.random.randn(mp, block_size) * 0.5
                W_block = np.random.randn(pn, block_size) * 0.5
                
                # Mask to relevant indices
                k_indices = [3*block + b for b in range(3)]
                V_mask = np.zeros_like(V_block)
                W_mask = np.zeros_like(W_block)
                for k in k_indices:
                    for j in range(m):
                        V_mask[j*p+k, :] = V_block[j*p+k, :]
                    for i in range(n):
                        W_mask[k*n+i, :] = W_block[k*n+i, :]
                
                U[start:end] = U_block
                V[start:end] = V_mask
                W[start:end] = W_mask
        else:
            # Random initialization
            scale = 0.5 / np.sqrt(target_rank)
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
        
        # Optimize using simple gradient descent with adaptive learning rate
        lr = 0.1
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
                
            # Compute residual R - T_target
            R = np.einsum('ar,br,cr->abc', U, V, W)
            residual = R - T_target
            loss = np.sum(residual**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_loss < tol:
                break
                
            # Compute gradients
            # dL/dU[a,r] = sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} residual[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} residual[a,b,c] * U[a,r] * V[b,r]
            
            grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.5
            elif loss < best_loss * 0.9:
                lr = min(lr * 1.1, 0.5)
        
        if best_loss < 1e-4:
            break
    
    if best_loss < 1e-4 and best_U is not None:
        # Snap to nearest 0.5
        U_s = np.round(best_U * 2) / 2
        V_s = np.round(best_V * 2) / 2
        W_s = np.round(best_W * 2) / 2
        
        # Verify
        if _verify_fast(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-9):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist(), "rank": target_rank}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Known upper bound: 54 (schoolbook)
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    
    # Try a few ranks near the lower bound
    for target_rank in range(lower_bound, min(schoolbook_rank, lower_bound + 20)):
        if time.time() > DEADLINE - 15:
            break
        result = _try_rank_optimize(n, m, p, target_rank, max_iters=100)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}

# EVOLVE-BLOCK-END
