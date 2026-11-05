"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T has indices (i*m+j, j*p+k, k*n+i) with value 1
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    
    # Build the full tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But tensor rank is typically much higher
    # Known: 4x4x4 has rank 49, so 4x4x5 might be around 50-60
    
    deadline = time.time() + 120.0
    best_rank = rank_schoolbook
    best_decomp = None
    
    # Schoolbook decomposition for fallback
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    # Start from rank 49 (known for 4x4x4) and go up
    # Actually, let's try a range of ranks
    
    def try_rank(r, max_iters=5000, lr=0.01, n_restarts=5):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        best_loss = np.inf
        best_params = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            loss_history = []
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor estimate
                # T_est[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
                # This is expensive, so compute loss efficiently
                
                # Compute loss = 0.5 * ||T_est - T||_F^2
                # grad_U[a,k] = sum_{b,c} V[b,k]*W[c,k]*(T_est[a,b,c] - T[a,b,c])
                # = sum_b V[b,k] * sum_c W[c,k]*(T_est[a,b,c] - T[a,b,c])
                
                # More efficient: 
                # Let E[a,b,c] = T_est[a,b,c] - T[a,b,c]
                # grad_U[a,k] = sum_{b,c} V[b,k]*W[c,k]*E[a,b,c]
                # = (V.T @ (E * W.T).sum(axis=2))[a,k] ... this is getting complex
                
                # Alternative: use einsum
                # T_est = np.einsum('ak,bk,ck->abc', U, V, W)
                # This is O(n*m * m*p * p*n * r) which is 16*20*20*r = 6400*r
                # For r=50, that's 320,000 operations per iteration - fine
                
                T_est = np.einsum('ak,bk,ck->abc', U, V, W)
                E = T_est - T
                loss = 0.5 * np.sum(E * E)
                loss_history.append(loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_params = (U.copy(), V.copy(), W.copy())
                
                if loss < 1e-10:
                    break
                
                # Gradients
                # grad_U[a,k] = sum_{b,c} V[b,k]*W[c,k]*E[a,b,c]
                # = einsum('bk,ck,abc->ak', V, W, E)
                # grad_V[b,k] = sum_{a,c} U[a,k]*W[c,k]*E[a,b,c]
                # = einsum('ak,ck,abc->bk', U, W, E)
                # grad_W[c,k] = sum_{a,b} U[a,k]*V[b,k]*E[a,b,c]
                # = einsum('ak,bk,abc->ck', U, V, E)
                
                grad_U = np.einsum('bk,ck,abc->ak', V, W, E)
                grad_V = np.einsum('ak,ck,abc->bk', U, W, E)
                grad_W = np.einsum('ak,bk,abc->ck', U, V, E)
                
                # Adaptive learning rate
                effective_lr = lr / (1.0 + it * 0.001)
                
                U -= effective_lr * grad_U
                V -= effective_lr * grad_V
                W -= effective_lr * grad_W
                
                # Check convergence
                if it > 100 and loss_history[-1] >= loss_history[-10]:
                    # Plateau, try larger step or break
                    if loss_history[-1] > best_loss * 1.01:
                        break
        
        if best_params is not None and best_loss < 1e-8:
            return best_params
        return None
    
    # Try ranks from lower to higher
    # Based on known results, 4x4x4 has rank 49
    # For 4x4x5, let's try from 49 upward
    # But we should also try lower values in case there's a clever construction
    
    # Actually, let's be more strategic. Try a few key ranks.
    candidate_ranks = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, max_iters=3000, lr=0.02, n_restarts=3)
        if result is not None:
            U, V, W = result
            # Verify
            T_est = np.einsum('ak,bk,ck->abc', U, V, W)
            if np.max(np.abs(T_est - T)) < 1e-8:
                best_rank = r
                best_decomp = (U, V, W)
                break
    
    if best_decomp is None:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
