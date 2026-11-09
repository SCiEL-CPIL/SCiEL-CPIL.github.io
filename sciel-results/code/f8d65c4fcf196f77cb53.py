import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build the target tensor T as a 3D array
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition as fallback (Rank 63)
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
        # T_rec[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Using einsum for speed
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def optimize_rank(R, max_iters=20000, lr=0.05, seed=42):
        """
        Try to find a rank-R decomposition using gradient descent.
        Initialization strategy:
        - If R >= 57, use block-structured initialization (27+27+3).
        - Else, use random initialization.
        """
        if time.time() > deadline:
            return None
        
        np.random.seed(seed)
        
        # Initialization
        if R >= 57:
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            # Block 1: 3x3x3 (cols 0-26)
            # Block 2: 3x3x3 (cols 27-53)
            # Block 3: 3x3x1 (cols 54-56)
            
            # Helper to fill a 3x3x3 block starting at column offset
            def fill_3x3_block(offset, k_offset):
                col = offset
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            U[i * 3 + j, col] = 1
                            V[j * 3 + (k + k_offset), col] = 1
                            W[(k + k_offset) * 3 + i, col] = 1
                            col += 1
            
            fill_3x3_block(0, 0)
            fill_3x3_block(27, 3)
            
            # Block 3: 3x3x1 (k=6)
            col = 54
            for i in range(3):
                for j in range(3):
                    U[i * 3 + j, col] = 1
                    V[j * 3 + 6, col] = 1
                    W[6 * 3 + i, col] = 1
                    col += 1
            
            # If R > 57, add random columns
            if R > 57:
                U[:, 57:] = np.random.randn(nm, R - 57) * 0.1
                V[:, 57:] = np.random.randn(mp, R - 57) * 0.1
                W[:, 57:] = np.random.randn(pn, R - 57) * 0.1
        else:
            # Random initialization for lower ranks
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
        
        # Gradient Descent
        # Loss L = 0.5 * sum (T_rec - T)^2
        # grad_U[a, r] = sum_{b,c} (T_rec[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
        # This can be computed as:
        # grad_U = einsum('abc,br,cr->ar', error, V, W)
        # grad_V = einsum('abc,ar,cr->br', error, U, W)
        # grad_W = einsum('abc,ar,br->cr', error, U, V)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Forward pass
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            
            # Check convergence
            if np.max(np.abs(error)) < 1e-8:
                return (U, V, W)
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Adaptive learning rate reduction if stuck
            if it % 1000 == 0 and it > 0:
                # Simple heuristic: if error didn't drop, reduce lr
                pass # Skipping complex adaptive logic for brevity, fixed lr is often enough for small tensors if init is good

        # Final check
        if check_decomposition(U, V, W):
            return (U, V, W)
            
        return None

    best_rank = 63
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try ranks from 40 to 62
    # We prioritize lower ranks
    for R in range(40, 63):
        if time.time() > deadline - 10.0:
            break
        
        # Try a few seeds for robustness
        found = None
        for seed in [42, 123, 777]:
            if time.time() > deadline - 10.0:
                break
            result = optimize_rank(R, max_iters=5000, lr=0.05, seed=seed)
            if result is not None:
                found = result
                break
        
        if found is not None:
            best_rank = R
            best_U, best_V, best_W = found
            break # Found a valid one, stop searching lower ranks? No, we want the MINIMAL.
            # Actually, if we found rank R, we should try R-1, R-2...
            # The loop goes 40 to 62. So if we find 45, we should continue to 44, 43...
            # But the loop is increasing. So we should iterate from high to low?
            # No, we want the LOWEST rank. So we should iterate from LOW to HIGH?
            # If we find rank 40, we are done.
            # If we don't find 40, we try 41, etc.
            # So the loop `for R in range(40, 63)` is correct.
            # If we find a solution for R, we break because it's the lowest we've tried so far.
            # Wait, if we start at 40 and fail, we try 41. If we succeed at 41, we break.
            # This gives the minimal rank found in the range.
            
    # Convert to list format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
