import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Build the target tensor
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    start_time = time.time()
    deadline = 135.0  # 15s safety margin
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, target, atol=1e-9)
    
    def snap_to_half_int(arr):
        """Snap array to nearest half-integer."""
        return np.round(arr * 2.0) / 2.0
    
    def tensor_residual(U, V, W):
        """Compute ||target - U*V*W||_F efficiently."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(target - T)
    
    def get_schoolbook():
        """Return schoolbook decomposition (rank = n*m*p = 50)."""
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find a decomposition with rank < 50
    best_result = None
    
    # Strategy 1: Try ranks from 49 down to 35 with gradient descent (fixed: no inner r-loop)
    for target_rank in range(49, 35, -1):
        if time.time() > deadline:
            break
        
        found = False
        n_restarts = 30
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart)
            
            # Initialize with small random half-integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_u, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_v, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_w, target_rank))
            
            # Gradient descent with half-integer projection (ONE update per iteration, not per-r)
            for iteration in range(40):
                if time.time() > deadline:
                    break
                
                # Compute residual
                T_current = np.einsum('ar,br,cr->abc', U, V, W)
                diff = target - T_current
                
                # Check convergence
                if np.allclose(T_current, target, atol=1e-9):
                    found = True
                    best_result = (U.copy(), V.copy(), W.copy(), target_rank)
                    break
                
                # Compute gradients ONCE
                grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Adaptive learning rate
                norm_grad = np.sqrt(np.sum(grad_u**2) + np.sum(grad_v**2) + np.sum(grad_w**2))
                if norm_grad < 1e-6:
                    break
                lr = 0.05 * (1.0 / (1.0 + norm_grad))
                
                # Take a step
                U -= lr * grad_u
                V -= lr * grad_v
                W -= lr * grad_w
                
                # Snap to half-integers
                U = snap_to_half_int(U)
                V = snap_to_half_int(V)
                W = snap_to_half_int(W)
            
            if found:
                break
        
        if found:
            break
    
    # Strategy 2: If not found, try structured initialization based on Strassen-like patterns
    if best_result is None and time.time() < deadline:
        # Try various structured decompositions
        for target_rank in range(49, 35, -1):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(20):
                if time.time() > deadline:
                    break
                
                np.random.seed(3000 + restart)
                
                # Initialize with a mix of structured and random
                U = np.zeros((dim_u, target_rank))
                V = np.zeros((dim_v, target_rank))
                W = np.zeros((dim_w, target_rank))
                
                # Fill first few columns with schoolbook-like patterns
                for col in range(min(target_rank, 25)):
                    i = col // 25
                    j = (col // 5) % 5
                    k = col % 5
                    if i < n and j < m and k < p:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                
                # Add random perturbation to remaining columns
                for col in range(min(target_rank, 25), target_rank):
                    U[:, col] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_u)
                    V[:, col] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_v)
                    W[:, col] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=dim_w)
                
                # Optimize with gradient descent
                for iteration in range(30):
                    if time.time() > deadline:
                        break
                    
                    T_current = np.einsum('ar,br,cr->abc', U, V, W)
                    diff = target - T_current
                    
                    if np.allclose(T_current, target, atol=1e-9):
                        found = True
                        best_result = (U.copy(), V.copy(), W.copy(), target_rank)
                        break
                    
                    grad_u = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                    grad_v = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                    grad_w = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                    
                    norm_grad = np.sqrt(np.sum(grad_u**2) + np.sum(grad_v**2) + np.sum(grad_w**2))
                    if norm_grad < 1e-6:
                        break
                    lr = 0.03 * (1.0 / (1.0 + norm_grad))
                    
                    U -= lr * grad_u
                    V -= lr * grad_v
                    W -= lr * grad_w
                    
                    U = snap_to_half_int(U)
                    V = snap_to_half_int(V)
                    W = snap_to_half_int(W)
                
                if found:
                    break
            
            if found:
                break
    
    # Return the best result found, or schoolbook as fallback
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
