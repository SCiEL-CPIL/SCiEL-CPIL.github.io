"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 130  # 130 second budget (10s safety margin)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + l, col] = 1.0
                    W[l * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Get schoolbook solution as baseline
    U0, V0, W0 = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    def compute_error_fast(U, V, W):
        """Compute Frobenius norm error using numpy einsum."""
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        total = np.zeros((n, m, m, p))
        for r in range(rank):
            contrib = np.einsum('ij,kl,li->ijkl', U_r[:, :, r], V_r[:, :, r], W_r[:, :, r])
            total += contrib
        
        target = np.zeros((n, m, m, p))
        for j in range(m):
            target[:, j, j, :] = 1.0
        
        return np.linalg.norm(total - target)
    
    def try_rank(rank, max_trials=3, max_iters=200):
        """Try to find a valid decomposition at given rank using gradient descent."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for trial in range(max_trials):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            lr = 0.01
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                err = compute_error_fast(U, V, W)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    return True, U, V, W
                
                # Compute gradients via finite differences (expensive but simple)
                # Use a few random directions for gradient estimation
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Estimate gradients using a few random perturbations
                for _ in range(3):
                    dU = np.random.randn(*U.shape) * 0.01
                    dV = np.random.randn(*V.shape) * 0.01
                    dW = np.random.randn(*W.shape) * 0.01
                    
                    err_plus = compute_error_fast(U + dU, V + dV, W + dW)
                    err_minus = compute_error_fast(U - dU, V - dV, W - dW)
                    
                    grad_U += (err_plus - err_minus) / (2 * 0.01) * dU
                    grad_V += (err_plus - err_minus) / (2 * 0.01) * dV
                    grad_W += (err_plus - err_minus) / (2 * 0.01) * dW
                
                grad_U /= 3
                grad_V /= 3
                grad_W /= 3
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                lr *= 0.99
            
            if best_err < 1e-10:
                return True, best_U, best_V, best_W
        
        return False, best_U, best_V, best_W
    
    # Try ranks from 50 up to 71
    for rank in range(50, 72):
        if time.time() > deadline - 30:
            break
        
        success, U, V, W = try_rank(rank, max_trials=2, max_iters=100)
        if success:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If we found a better rank, use it; otherwise return schoolbook
    if best_rank < n * m * p:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = U0, V0, W0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
