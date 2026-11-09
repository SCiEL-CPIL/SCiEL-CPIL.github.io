import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def _compute_error(U, V, W, n, m, p):
    """Compute the squared error of the decomposition."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # We can do this by iterating over k or using einsum
    # T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    # This creates a (n, m, p) tensor.
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    err = np.sum((T - 1.0) ** 2)
    return err

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients for U, V, W."""
    rank = U.shape[1]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    residual = T - 1.0
    
    # dE/dU_ijc = 2 * sum_k residual[i,j,k] * V_jkc * W_kic
    # dU_reshaped[i, j, c] = 2 * sum_k residual[i, j, k] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
    # Using einsum: 'ijk,jkc,kic->ijc'
    dU_reshaped = 2.0 * np.einsum('ijk,jkc,kic->ijc', residual, V_reshaped, W_reshaped)
    
    # dE/dV_jkc = 2 * sum_i residual[i,j,k] * U_ijc * W_kic
    # dV_reshaped[j, k, c] = 2 * sum_i residual[i, j, k] * U_reshaped[i, j, c] * W_reshaped[k, i, c]
    # Using einsum: 'ijk,ijc,kic->jkc'
    dV_reshaped = 2.0 * np.einsum('ijk,ijc,kic->jkc', residual, U_reshaped, W_reshaped)
    
    # dE/dW_kic = 2 * sum_j residual[i,j,k] * U_ijc * V_jkc
    # dW_reshaped[k, i, c] = 2 * sum_j residual[i, j, k] * U_reshaped[i, j, c] * V_reshaped[j, k, c]
    # Using einsum: 'ijk,ijc,jkc->kic'
    dW_reshaped = 2.0 * np.einsum('ijk,ijc,jkc->kic', residual, U_reshaped, V_reshaped)
    
    return dU_reshaped.reshape(n*m, rank), dV_reshaped.reshape(m*p, rank), dW_reshaped.reshape(p*n, rank)

def _optimize(U, V, W, n, m, p, max_iters=1000, tol=1e-12, lr=1e-3):
    """Gradient descent optimization."""
    for i in range(max_iters):
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
        
        dU, dV, dW = _compute_gradients(U, V, W, n, m, p)
        
        # Normalize gradients to avoid exploding steps
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm == 0:
            break
        dU /= grad_norm
        dV /= grad_norm
        dW /= grad_norm
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Adaptive learning rate (simple)
        if i % 100 == 0:
            lr *= 0.9
            if lr < 1e-6:
                lr = 1e-6
                
    return U, V, W, _compute_error(U, V, W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140 # Safety margin
    
    # Start with trivial rank
    current_rank = n * m * p
    U = np.zeros((n*m, current_rank))
    V = np.zeros((m*p, current_rank))
    W = np.zeros((p*n, current_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Optimize the trivial solution to ensure it's perfect (it should be, but just in case)
    U, V, W, err = _optimize(U, V, W, n, m, p, max_iters=100, tol=1e-10)
    
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank
    # We iterate downwards.
    # For each rank, we try to find a solution.
    # We initialize by taking the best solution of the previous rank and "merging" or truncating.
    # A simple strategy: keep the best 'r' columns from the 'r+1' solution?
    # Or better: re-initialize randomly but use the previous solution as a warm start if possible.
    
    # Since the problem is small, we can try ranks from a lower bound up to the current best.
    # Lower bound is roughly max(n, m, p) = 6.
    # But let's start from current_rank - 1 and go down.
    
    # To save time, we will try to reduce rank by 1 in each step.
    # If we fail to reduce, we stop.
    
    for target_rank in range(current_rank - 1, 5, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this target rank
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            if attempt == 0:
                # Warm start: take the best solution from previous rank and truncate
                # This might not work directly because the columns are not ordered by importance.
                # Instead, we can try to project the previous solution onto a lower rank space?
                # Or just use the previous solution's U, V, W but set the last column to 0 and optimize?
                # Let's try: take best_U, best_V, best_W, zero out the last column, and optimize.
                U_init = best_U.copy()
                V_init = best_V.copy()
                W_init = best_W.copy()
                U_init[:, -1] = 0
                V_init[:, -1] = 0
                W_init[:, -1] = 0
                # Actually, we need to reduce the dimension.
                # We can just take the first 'target_rank' columns? No, they are arbitrary.
                # Let's try a random initialization for attempts > 0.
                # For attempt 0, let's try a "greedy" reduction:
                # Compute the norm of each column in the previous solution.
                # Keep the top 'target_rank' columns.
                norms = np.linalg.norm(best_U, axis=0)
                idx = np.argsort(norms)[::-1][:target_rank]
                U_init = best_U[:, idx]
                V_init = best_V[:, idx]
                W_init = best_W[:, idx]
            else:
                # Random initialization
                rng = np.random.RandomState(attempt * 100 + target_rank)
                U_init = rng.randn(n*m, target_rank)
                V_init = rng.randn(m*p, target_rank)
                W_init = rng.randn(p*n, target_rank)
                
            U_opt, V_opt, W_opt, err = _optimize(U_init, V_init, W_init, n, m, p, max_iters=500, tol=1e-10, lr=1e-2)
            
            if err < 1e-8:
                # Check for exactness
                if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    found = True
                    break # Move to next lower rank
                    
        if not found:
            # If we couldn't reduce the rank, we might still be able to go lower with more attempts?
            # Or maybe we should stop?
            # Let's try a few more ranks down if we have time.
            pass

    # Final verification of the best found
    if best_rank < n * m * p:
        if _verify_decomposition(best_U, best_V, best_W, n, m, p):
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
            
    # Fallback to trivial
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-8:
                    return False
    return True
