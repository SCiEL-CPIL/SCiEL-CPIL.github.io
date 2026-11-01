"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication.
    
    Uses structured optimization with sum-of-squares loss, adaptive learning rates,
    and a final polish pass to find lower-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    
    # Build the full tensor T[a, b, c] where:
    #   a = i*m + j  (row of A, col of A)
    #   b = j*p + k  (row of B, col of B)
    #   c = k*n + i  (output, trace convention: k*n + i)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 50
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Verify schoolbook
    Rec_sb = np.einsum('ik,jk,lk->ijl', U_sb, V_sb, W_sb)
    err_sb = np.max(np.abs(T - Rec_sb))
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find lower-rank decompositions
    # Use sum-of-squares loss for smoother optimization
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 10), -1):
        if time.time() > deadline - 15:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: start from a perturbed version of schoolbook
            # Take a subset of schoolbook columns and add random noise
            if attempt == 0:
                # Use first target_rank columns of schoolbook, scaled
                U = U_sb[:, :target_rank].copy()
                V = V_sb[:, :target_rank].copy()
                W = W_sb[:, :target_rank].copy()
                # Add small random perturbation
                U += np.random.randn(n * m, target_rank) * 0.01
                V += np.random.randn(m * p, target_rank) * 0.01
                W += np.random.randn(p * n, target_rank) * 0.01
            elif attempt == 1:
                # Random initialization with moderate scale
                U = np.random.randn(n * m, target_rank) * 0.3
                V = np.random.randn(m * p, target_rank) * 0.3
                W = np.random.randn(p * n, target_rank) * 0.3
            elif attempt == 2:
                # Sparse initialization
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for k in range(target_rank):
                    # Randomly assign non-zero entries
                    idx_u = np.random.choice(n * m, size=min(5, n * m), replace=False)
                    idx_v = np.random.choice(m * p, size=min(5, m * p), replace=False)
                    idx_w = np.random.choice(p * n, size=min(5, p * n), replace=False)
                    U[idx_u, k] = 1.0
                    V[idx_v, k] = 1.0
                    W[idx_w, k] = 1.0
            else:
                # Random with different scale
                scale = 0.1 * (attempt - 2)
                U = np.random.randn(n * m, target_rank) * scale
                V = np.random.randn(m * p, target_rank) * scale
                W = np.random.randn(p * n, target_rank) * scale
            
            # Optimize using sum-of-squares loss with adaptive learning rate
            lr = 0.05
            lr_decay = 0.995
            best_err = np.inf
            best_U_iter = U.copy()
            best_V_iter = V.copy()
            best_W_iter = W.copy()
            
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
                diff = T - Rec
                loss = np.sum(diff * diff)
                
                if loss < 1e-10:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
                
                if loss < best_err:
                    best_err = loss
                    best_U_iter = U.copy()
                    best_V_iter = V.copy()
                    best_W_iter = W.copy()
                
                # Compute gradients for sum-of-squares loss
                # dL/dU[i,k] = -2 * sum_{j,l} diff[i,j,l] * V[j,k] * W[l,k]
                # dL/dV[j,k] = -2 * sum_{i,l} diff[i,j,l] * U[i,k] * W[l,k]
                # dL/dW[l,k] = -2 * sum_{i,j} diff[i,j,l] * U[i,k] * V[j,k]
                
                grad_U = 2 * np.einsum('ijl,jk,lk->ik', diff, V, W)
                grad_V = 2 * np.einsum('ijl,ik,lk->jk', diff, U, W)
                grad_W = 2 * np.einsum('ijl,ik,jk->lk', diff, U, V)
                
                # Adaptive step: reduce if loss increases
                U_new = U - lr * grad_U
                V_new = V - lr * grad_V
                W_new = W - lr * grad_W
                
                Rec_new = np.einsum('ik,jk,lk->ijl', U_new, V_new, W_new)
                diff_new = T - Rec_new
                loss_new = np.sum(diff_new * diff_new)
                
                if loss_new < loss:
                    U, V, W = U_new, V_new, W_new
                else:
                    lr *= 0.5
                
                lr *= lr_decay
            
            # Check final error
            Rec = np.einsum('ik,jk,lk->ijl', best_U_iter, best_V_iter, best_W_iter)
            err = np.max(np.abs(T - Rec))
            if err < 1e-8:
                best_rank = target_rank
                best_U = best_U_iter
                best_V = best_V_iter
                best_W = best_W_iter
                break
    
    # Final polish pass on the best solution found
    if best_rank < rank_schoolbook and time.time() < deadline - 3:
        U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
        lr = 0.01
        for step in range(50):
            if time.time() > deadline - 1:
                break
            Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
            diff = T - Rec
            loss = np.sum(diff * diff)
            if loss < 1e-12:
                break
            grad_U = 2 * np.einsum('ijl,jk,lk->ik', diff, V, W)
            grad_V = 2 * np.einsum('ijl,ik,lk->jk', diff, U, W)
            grad_W = 2 * np.einsum('ijl,ik,jk->lk', diff, U, V)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            lr *= 0.98
        
        Rec = np.einsum('ik,jk,lk->ijl', U, V, W)
        err = np.max(np.abs(T - Rec))
        if err < 1e-8:
            best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
