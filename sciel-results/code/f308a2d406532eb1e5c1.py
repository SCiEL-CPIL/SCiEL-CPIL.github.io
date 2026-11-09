import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _try_structured_decomposition(n, m, p):
    """
    Try to find a low-rank decomposition using the structure of the tensor.
    For 2x5x6, exploit that n=2 is small.
    """
    # Strategy: Use the fact that for each (i,k), the sum over j of A[i,j]*B[j,k] = C[i,k]
    # We can try to use a decomposition where W depends only on (k,i) in a simple way.
    
    # Try rank = m + p - 1 = 10? No, that's too optimistic.
    # For 2x5x6, known upper bound is around 55-58.
    
    # Let's try a different approach: use the "split" method
    # Partition the middle dimension j into groups
    
    # Actually, let's try a direct construction:
    # Set W[k,i,r] = 1 for r < p, and 0 otherwise, for i=0
    # Set W[k,i,r] = 0 for r < p, and 1 for r in [p, 2p) for i=1
    # This won't work directly.
    
    # Better: Use the identity that 1 = sum_{r=1}^{R} u_r * v_r * w_r
    # where we can choose the factors to be sparse.
    
    # For now, try a few specific ranks with random but structured init
    for target_rank in [55, 50, 45]:
        if target_rank < 30:
            break
        
        # Initialize with a structured pattern
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Fill in a basic pattern
        for r in range(target_rank):
            # Assign each rank-1 term to cover certain (i,j,k) triples
            i = r % n
            j = (r // n) % m
            k = (r // (n*m)) % p
            
            U[i*m + j, r] = 1.0
            V[j*p + k, r] = 1.0
            W[k*n + i, r] = 1.0
        
        # This is just schoolbook for rank < n*m*p, won't work
        # Need a smarter approach
        
    return None

def _optimize_with_structure(n, m, p, target_rank, time_limit, max_restarts=5):
    """
    Try to find a decomposition using optimization with better initialization.
    """
    deadline = time.time() + time_limit
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Better initialization: use a mixture of structured and random
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Add a structured component
        for r in range(min(target_rank, n*m*p)):
            i = r % n
            j = (r // n) % m
            k = (r // (n*m)) % p
            U[i*m + j, r] += 1.0
            V[j*p + k, r] += 1.0
            W[k*n + i, r] += 1.0
        
        def cost(params):
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            U_arr = U_flat.reshape(n*m, target_rank)
            V_arr = V_flat.reshape(m*p, target_rank)
            W_arr = W_flat.reshape(p*n, target_rank)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                        err += (s - 1.0)**2
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use a simple gradient descent with adaptive step
        step_size = 0.01
        best_cost = cost(x0)
        
        for it in range(100):
            if time.time() > deadline:
                break
            
            # Compute gradient numerically (expensive but simple)
            # For speed, use finite differences on a subset
            grad = np.zeros_like(x0)
            eps = 1e-5
            
            # Sample some coordinates for gradient estimation
            num_samples = min(50, len(x0))
            indices = random.sample(range(len(x0)), num_samples)
            
            for idx in indices:
                x_plus = x0.copy()
                x_plus[idx] += eps
                x_minus = x0.copy()
                x_minus[idx] -= eps
                
                grad[idx] = (cost(x_plus) - cost(x_minus)) / (2 * eps)
            
            # Update
            x0 = x0 - step_size * grad
            new_cost = cost(x0)
            
            if new_cost < best_cost:
                best_cost = new_cost
                # Decay step size
                step_size *= 0.95
            else:
                step_size *= 0.5
        
        if best_cost < 1e-8:
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            U_final = x0[:size_U].reshape(n*m, target_rank)
            V_final = x0[size_U:size_U+size_V].reshape(m*p, target_rank)
            W_final = x0[size_U+size_V:].reshape(p*n, target_rank)
            
            if _verify(U_final, V_final, W_final, n, m, p):
                return U_final, V_final, W_final
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    for target_rank in range(best_rank - 1, 40, -1):
        if time.time() > deadline - 15:
            break
        
        result = _optimize_with_structure(n, m, p, target_rank, deadline - 20, max_restarts=3)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
