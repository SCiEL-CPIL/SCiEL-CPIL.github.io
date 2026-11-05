"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    DEADLINE = time.time() + 140.0  # 140s out of 150s budget
    START_TIME = time.time()
    
    # Return a valid schoolbook decomposition if any error occurs
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_fast(U, V, W, tol=1e-10):
        """Fast verification using numpy vectorization."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Build tensor via einsum: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Use einsum for efficiency
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if T[i,j,k] = 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=tol):
            return False
        return True
    
    def compute_residual(U, V, W):
        """Compute sum of squared residuals."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.sum((T - 1.0) ** 2)
    
    def structured_init(r, seed=None):
        """
        Initialize using tensor structure.
        For 3x3x8, exploit that the left multiplication (3x3) has rank-3 structure.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Strategy: Use Kronecker product structure
        # U is (n*m, r) = (9, r)
        # V is (m*p, r) = (24, r)  
        # W is (p*n, r) = (24, r)
        
        # Initialize with structured random values
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        # Add some structure: make first few columns have pattern
        for c in range(min(3, r)):
            # Left multiplication structure
            for i in range(n):
                for j in range(m):
                    U[i * m + j, c] += 0.5 * (1 if i == j else 0.1)
            
            # Right multiplication structure  
            for j in range(m):
                for k in range(p):
                    V[j * p + k, c] += 0.5 * (1 if j == 0 else 0.1)
            
            for k in range(p):
                for i in range(n):
                    W[k * n + i, c] += 0.5 * (1 if k == 0 else 0.1)
        
        return U, V, W
    
    def compute_grads(U, V, W):
        """Compute gradients of residual w.r.t. U, V, W using chain rule."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # d(res)/dU[i,j,c] = 2*(T[i,j,k]-1) * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        residual = T - 1.0
        
        # Gradient for U: dU[i,j,c] = 2 * sum_k residual[i,j,k] * V[j,k,c] * W[k,i,c]
        grad_U = np.einsum('ijk,jkc,kic->ijc', residual, V_r, W_r) * 2
        # Gradient for V: dV[j,k,c] = 2 * sum_i residual[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V = np.einsum('ijk,ijc,kic->jkc', residual, U_r, W_r) * 2
        # Gradient for W: dW[k,i,c] = 2 * sum_j residual[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W = np.einsum('ijk,ijc,jkc->kic', residual, U_r, V_r) * 2
        
        return grad_U.reshape(n*m, rank), grad_V.reshape(m*p, rank), grad_W.reshape(p*n, rank)
    
    def gradient_descent(U, V, W, r, max_iters=1000, timeout=None, lr=0.01):
        """
        Gradient descent with adaptive learning rate.
        """
        if timeout is None:
            timeout = 30
        
        current_res = compute_residual(U, V, W)
        best_res = current_res
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        no_improve_count = 0
        lr_history = [lr]
        
        for it in range(max_iters):
            if time.time() > timeout:
                break
            
            if no_improve_count > 20:
                lr *= 0.7
                if lr < 1e-8:
                    break
                no_improve_count = 0
                lr_history.append(lr)
            
            # Compute gradients
            grad_U, grad_V, grad_W = compute_grads(U, V, W)
            
            # Gradient step
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_res = compute_residual(U, V, W)
            
            if new_res < current_res - 1e-12:
                current_res = new_res
                if new_res < best_res:
                    best_res = new_res
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve_count = 0
            else:
                # Revert
                U += lr * grad_U
                V += lr * grad_V
                W += lr * grad_W
                no_improve_count += 1
            
            if current_res < 1e-12:
                break
        
        return best_U, best_V, best_W, best_res
    
    def hybrid_search(U, V, W, r, timeout=None):
        """
        Hybrid: start with gradient descent, then switch to local search if stuck.
        """
        if timeout is None:
            timeout = 30
        
        # Phase 1: Gradient descent
        U, V, W, res = gradient_descent(U, V, W, r, max_iters=500, timeout=min(20, timeout * 0.6), lr=0.05)
        
        if res < 1e-10:
            return U, V, W, res
        
        # Phase 2: Local search to polish
        U, V, W, res = local_search(U, V, W, r, max_iters=200, timeout=min(10, timeout * 0.35))
        
        return U, V, W, res
    
    def local_search(U, V, W, r, max_iters=500, timeout=None):
        """
        Local search using gradient-free method with better structure awareness.
        """
        if timeout is None:
            timeout = 30
        
        current_res = compute_residual(U, V, W)
        best_res = current_res
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        step_size = 0.1
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > timeout:
                break
            
            if no_improve_count > 50:
                step_size *= 0.9
                if step_size < 0.001:
                    break
                no_improve_count = 0
            
            # Try multiple random perturbations
            improved = False
            for _ in range(10):
                c = random.randint(0, r - 1)
                
                # Choose which matrix to perturb
                matrix_choice = random.randint(0, 2)
                
                if matrix_choice == 0:
                    idx = random.randint(0, n * m - 1)
                    d = random.choice([-step_size, 0, step_size])
                    if d == 0:
                        continue
                    U[idx, c] += d
                    new_res = compute_residual(U, V, W)
                    U[idx, c] -= d
                elif matrix_choice == 1:
                    idx = random.randint(0, m * p - 1)
                    d = random.choice([-step_size, 0, step_size])
                    if d == 0:
                        continue
                    V[idx, c] += d
                    new_res = compute_residual(U, V, W)
                    V[idx, c] -= d
                else:
                    idx = random.randint(0, p * n - 1)
                    d = random.choice([-step_size, 0, step_size])
                    if d == 0:
                        continue
                    W[idx, c] += d
                    new_res = compute_residual(U, V, W)
                    W[idx, c] -= d
                
                if new_res < current_res - 1e-12:
                    # Accept the move
                    if matrix_choice == 0:
                        U[idx, c] += d
                    elif matrix_choice == 1:
                        V[idx, c] += d
                    else:
                        W[idx, c] += d
                    current_res = new_res
                    improved = True
                    
                    if new_res < best_res:
                        best_res = new_res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if not improved:
                no_improve_count += 1
            
            if current_res < 1e-12:
                break
        
        return best_U, best_V, best_W, best_res
    
    def try_rank(r, num_trials=30, timeout=None):
        """Try to find a rank-r decomposition."""
        if timeout is None:
            timeout = 40
        
        for trial in range(num_trials):
            if time.time() > timeout:
                break
            
            # Initialize
            U, V, W = structured_init(r, seed=trial * 1000 + 42)
            
            # Hybrid search
            U, V, W, res = hybrid_search(U, V, W, r, timeout=min(25, timeout - (time.time() - START_TIME)))
            
            if res < 1e-10:
                if verify_fast(U.tolist(), V.tolist(), W.tolist()):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
        return None
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But tensor structure may require more.
    # For 3x3x8, try ranks from 24 to 72
    
    best_result = None
    trial_ranks = [24, 28, 32, 36, 40, 48, 56, 64, 72]
    
    for r in trial_ranks:
        if time.time() > DEADLINE - 20:
            break
        
        # Limit time per rank attempt
        time_per_rank = min(25, max(8, (DEADLINE - time.time()) / 2.0 - 20))
        
        result = try_rank(r, num_trials=20, timeout=time_per_rank)
        
        if result is not None:
            best_result = result
            print(f"Found rank-{r} decomposition!")
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    print("Falling back to schoolbook decomposition")
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
