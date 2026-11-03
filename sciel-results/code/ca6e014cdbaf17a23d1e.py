"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    # Helper to check and snap
    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Snap to nearest half-integer
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        # Reconstruct
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        # Check exactness
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        return {
            "U": U_snap.tolist(),
            "V": V_snap.tolist(),
            "W": W_snap.tolist()
        }

    # Fast custom Gradient Descent for Tensor Decomposition
    def fast_gradient_descent(U, V, W, r, max_iters=5000, tol=1e-10):
        """
        Minimizes || T_target - sum_r U[:,r] V[:,r]^T W[:,r]^T ||_F^2
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        lr = 0.01
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        patience = 0
        patience_limit = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute Reconstruction
            # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_recon - T_target
            loss = np.sum(diff ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 0
            else:
                patience += 1
                
            if loss < tol:
                break
                
            if patience > patience_limit:
                break
                
            # Compute Gradients
            # dL/dU[a,r] = sum_{b,c} 2 * diff[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} 2 * diff[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} 2 * diff[a,b,c] * U[a,r] * V[b,r]
            
            # Efficient gradient calculation using einsum
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate: if loss increases, reduce lr
            if not np.isfinite(loss):
                break
                
            # Periodic lr reduction if stuck
            if it > 100 and it % 100 == 0 and loss > best_loss * 0.99:
                lr *= 0.95

        return best_U, best_V, best_W, best_loss

    # Fallback Schoolbook Rank 50 Solution
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Strategy: Try to find lower rank by removing terms and optimizing
    # We'll try ranks from 49 down to 40, with multiple seeds
    
    ranks_to_try = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple initializations
        for seed in [42, 123, 999, 111, 222, 333, 444, 555]:
            if time.time() > deadline - 5:
                break
                
            np.random.seed(seed)
            
            # Initialize with perturbation from schoolbook
            # Take first r columns of schoolbook and add noise
            if r <= rank_school:
                U_init = U_school[:, :r].copy()
                V_init = V_school[:, :r].copy()
                W_init = W_school[:, :r].copy()
            else:
                U_init = np.random.randn(n * m, r) * 0.5
                V_init = np.random.randn(m * p, r) * 0.5
                W_init = np.random.randn(p * n, r) * 0.5
            
            # Add small noise to help escape local minima
            noise_scale = 0.05
            U_init = U_init + np.random.randn(*U_init.shape) * noise_scale
            V_init = V_init + np.random.randn(*V_init.shape) * noise_scale
            W_init = W_init + np.random.randn(*W_init.shape) * noise_scale
            
            # Run gradient descent
            U_sol, V_sol, W_sol, final_loss = fast_gradient_descent(U_init, V_init, W_init, r, max_iters=3000)
            
            if final_loss < 1e-6:
                result = check_and_snap(U_sol, V_sol, W_sol, r)
                if result:
                    best_result = result
                    best_rank = r
                    found = True
                    break
        
        if found:
            break

    # If we found a better rank, also try to refine it
    if best_rank < rank_school:
        # Try to reduce further with the found solution as starting point
        for r_try in range(best_rank - 1, max(best_rank - 5, 30), -1):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(777)
            U_init = np.array(best_result["U"])[:, :r_try].copy()
            V_init = np.array(best_result["V"])[:, :r_try].copy()
            W_init = np.array(best_result["W"])[:, :r_try].copy()
            
            U_sol, V_sol, W_sol, final_loss = fast_gradient_descent(U_init, V_init, W_init, r_try, max_iters=2000)
            
            if final_loss < 1e-6:
                result = check_and_snap(U_sol, V_sol, W_sol, r_try)
                if result:
                    best_result = result
                    best_rank = r_try
                    break

    return best_result
# EVOLVE-BLOCK-END
