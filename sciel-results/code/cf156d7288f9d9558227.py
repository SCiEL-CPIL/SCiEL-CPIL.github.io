import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s
    
    n, m, p = 2, 2, 8
    
    # First, get the rank-28 Strassen solution as fallback
    fallback = _strassen_28()
    
    # Try to find a lower-rank decomposition via local search from Strassen
    for target_rank in [24, 25, 26, 27]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _search_local(target_rank, remaining - 2)
        if result is not None:
            return result
            
    return fallback

def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_strassen = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    num_blocks = p // 2
    col = 0
    for t in range(num_blocks):
        for r in range(7):
            for i in range(4):
                U[i, col] = U_strassen[r][i]
            v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
            for l in range(4):
                V[v_indices[l], col] = V_strassen[r][l]
            w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for l in range(4):
                W[w_indices[l], col] = W_strassen[r][l]
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_local(target_rank, time_budget):
    start = time.time()
    n, m, p = 2, 2, 8
    num_a = n * m  # 4
    num_b = m * p  # 16
    num_c = p * n  # 16
    
    # Build the target tensor T
    T = np.zeros((num_a, num_b, num_c), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(8):
                a = i * 2 + j
                b = j * 8 + k
                c = k * 2 + i
                T[a, b, c] = 1.0
                
    # Get the rank-28 Strassen solution
    sol_28 = _strassen_28()
    U_28 = np.array(sol_28["U"])
    V_28 = np.array(sol_28["V"])
    W_28 = np.array(sol_28["W"])
    
    # We will try to remove (28 - target_rank) columns and optimize the rest
    num_to_remove = 28 - target_rank
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() - start > time_budget:
            break
            
        # Randomly select columns to keep
        cols_to_keep = np.random.choice(28, target_rank, replace=False)
        cols_to_keep.sort()
        
        U = U_28[:, cols_to_keep].copy()
        V = V_28[:, cols_to_keep].copy()
        W = W_28[:, cols_to_keep].copy()
        
        # Optimize U, V, W to minimize || sum_r U_r V_r W_r^T - T ||_F
        # Using a simple gradient descent or alternating minimization
        # Since dimensions are small, we can use a simple iterative method
        
        # Initialize with small random perturbations to escape local minima
        # But for rank reduction, we might need to move significantly
        # Let's try a few steps of gradient descent on the flattened variables
        
        # Residual R = T - T_hat
        # T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # We will use a simple iterative scheme:
        # For each r, update U[:,r], V[:,r], W[:,r] to minimize the residual
        # This is similar to ALS (Alternating Least Squares) but for tensor rank
        
        # Actually, a better approach for small rank is to use the fact that
        # the tensor is sparse.
        
        # Let's just do a random search in the neighborhood of the Strassen solution
        # with random scaling and shifts, then round.
        
        # Perturb the kept columns
        noise = np.random.normal(0, 0.1, size=U.shape)
        U = U + noise
        noise = np.random.normal(0, 0.1, size=V.shape)
        V = V + noise
        noise = np.random.normal(0, 0.1, size=W.shape)
        W = W + noise
        
        # Try to optimize for a few iterations
        # Simple gradient step
        # T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        # This is expensive to compute every step if done naively, but 4x16x16 is small.
        
        # Let's do 20 iterations of a simple update
        for _ in range(20):
            # Compute T_hat
            T_hat = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_hat
            
            # Gradient for U[:,r]: sum_{b,c} R[a,b,c] V[b,r] W[c,r]
            # We want to decrease the error, so U -= lr * grad
            # But we want to fit T, so U += lr * grad_R where R = T - T_hat
            # dR/da = -dT_hat/da = -V[b,r]W[c,r]
            # So grad_U[a,r] = sum_{b,c} -R[a,b,c] V[b,r] W[c,r]
            # Update: U[a,r] += lr * sum_{b,c} R[a,b,c] V[b,r] W[c,r]
            
            lr = 0.01
            for r in range(target_rank):
                # Grad U
                gU = np.einsum('abc,br,cr->a', R, V[:,r], W[:,r])
                U[:,r] += lr * gU
                
                # Grad V
                gV = np.einsum('abc,ar,cr->b', R, U[:,r], W[:,r])
                V[:,r] += lr * gV
                
                # Grad W
                gW = np.einsum('abc,ar,br->c', R, U[:,r], V[:,r])
                W[:,r] += lr * gW
            
            # Check convergence
            err = np.linalg.norm(R)
            if err < 1e-3:
                break
        
        # Check if we found a solution
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_hat)
        
        if err < 1e-2:
            # Round to half-integers
            U_int = np.round(U * 2) / 2
            V_int = np.round(V * 2) / 2
            W_int = np.round(W * 2) / 2
            
            # Verify exactness
            T_check = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
            if np.allclose(T_check, T, atol=1e-6):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    return None

def _search_low_rank(target_rank, time_budget):
    """Placeholder to keep interface if needed, but we use _search_local now."""
    return None
