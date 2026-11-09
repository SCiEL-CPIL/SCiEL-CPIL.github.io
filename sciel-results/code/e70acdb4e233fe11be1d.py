import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def compute_error(U, V, W):
        """Compute the error tensor."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return T_rec - T
    
    def try_gradient_descent(r, max_iters=2000, lr=0.05, seed=42, tol=1e-8):
        """Try to find a rank-r decomposition using gradient descent with adaptive learning rate."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        best_error = np.inf
        best_state = (U.copy(), V.copy(), W.copy())
        patience = 0
        max_patience = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            error = compute_error(U, V, W)
            err_norm = np.max(np.abs(error))
            
            if err_norm < best_error:
                best_error = err_norm
                best_state = (U.copy(), V.copy(), W.copy())
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                break
            
            if err_norm < tol:
                return (U, V, W)
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 1e-10:
                step = lr * (1.0 + np.log1p(err_norm))
                U -= step * grad_U
                V -= step * grad_V
                W -= step * grad_W
            else:
                break
        
        # Final check
        if check_decomposition(best_state[0], best_state[1], best_state[2]):
            return best_state
        
        # Try to polish with smaller steps
        U, V, W = best_state
        for _ in range(500):
            if time.time() > deadline:
                break
            error = compute_error(U, V, W)
            err_norm = np.max(np.abs(error))
            if err_norm < tol:
                return (U, V, W)
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r_idx in range(r):
                grad_U[:, r_idx] = np.einsum('abc,b,c->a', error, V[:, r_idx], W[:, r_idx])
                grad_V[:, r_idx] = np.einsum('abc,a,c->b', error, U[:, r_idx], W[:, r_idx])
                grad_W[:, r_idx] = np.einsum('abc,a,b->c', error, U[:, r_idx], V[:, r_idx])
            
            U -= 0.01 * grad_U
            V -= 0.01 * grad_V
            W -= 0.01 * grad_W
        
        if check_decomposition(U, V, W):
            return (U, V, W)
        
        return None
    
    def try_structured_search(r, max_attempts=500):
        """Try to find a rank-r decomposition with structured entries {0, 0.5, 1, -1}."""
        if time.time() > deadline:
            return None
        
        # For small ranks, try random structured initializations
        candidates = [0, 0.5, 1, -0.5, -1]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            U = np.random.choice(candidates, size=(nm, r))
            V = np.random.choice(candidates, size=(mp, r))
            W = np.random.choice(candidates, size=(pn, r))
            
            if check_decomposition(U, V, W):
                return (U, V, W)
            
            # Try gradient descent from this initialization
            result = try_gradient_descent(r, max_iters=100, lr=0.02, seed=attempt + 1000)
            if result is not None:
                return result
        
        return None
    
    # Try to find a lower rank decomposition
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # First try structured search for ranks significantly below 63
    # Based on Strassen-like ideas, we might be able to do better than n*m*p
    # For 3x3x3, best known is 23 (vs 27 schoolbook)
    # For 3x3x7, let's try ranks from 55 down to 45
    
    # Try gradient descent first for a few ranks
    ranks_to_try = [62, 60, 58, 56, 55, 54, 53, 52, 51, 50]
    
    for r in ranks_to_try:
        if time.time() > deadline - 30:  # Leave 30s for other checks
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 30:
                break
            
            result = try_gradient_descent(r, max_iters=500, lr=0.03, seed=seed)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
        
        if best_rank < rank_sb:
            break
    
    # If we haven't found anything, try structured search
    if best_rank == rank_sb:
        for r in [55, 54, 53, 52]:
            if time.time() > deadline - 20:
                break
            
            result = try_structured_search(r, max_attempts=200)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_result = (U, V, W)
                break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
