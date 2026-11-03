import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition (rank 100)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Construct target tensor for verification
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U_mat, V_mat, W_mat, T_target, tol=1e-6):
        """Check if U, V, W exactly decompose T_target"""
        R = U_mat.shape[1]
        # Compute tensor product: sum_r U[:,r] (x) V[:,r] (x) W[:,r]
        # This is expensive, so we check a few random indices or use efficient computation
        # Efficient way: Reshape and use einsum or broadcasting
        # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can check by computing the full tensor if R is small enough
        # For R=100, 20*25*20 = 10000 elements, sum over 100 = 1M ops, feasible
        
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        diff = np.max(np.abs(T_approx - T_target))
        return diff < tol, diff

    # Verify initial schoolbook
    is_valid, err = verify(U, V, W, T)
    if not is_valid:
        # Should not happen, but return anyway
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = rank_sb
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds limit
    
    # Try to reduce rank by removing linearly dependent columns
    # The tensor rank is the minimal number of rank-1 tensors needed.
    # If columns of U, V, W are linearly dependent in a certain way, we might reduce rank.
    # However, the schoolbook decomposition is typically minimal for this structure unless there's symmetry.
    
    # Strategy: Try to find a lower rank decomposition using random initialization and gradient descent
    # on the tensor factorization problem.
    
    # Since exact integer/half-integer solutions are required, we will:
    # 1. Try to find a real-valued solution with lower rank.
    # 2. Round to nearest half-integer.
    # 3. Verify exactness.
    
    # Given the difficulty of finding a lower rank decomposition for 4x5x5 (known to be 100? or less?),
    # and the time constraint, we will attempt a few random searches for lower ranks.
    
    # Known results:
    # 2x2x2: rank 7 (Strassen)
    # 3x3x3: rank 23 (Landsberg)
    # 4x4x4: rank 49 (Landsberg)
    # 5x5x5: rank 100? (Schoolbook is 125, but Strassen-like might be lower)
    # For 4x5x5, the schoolbook is 100.
    # Can we do better?
    # If we split 4x5 into 4x4 + 4x1, and 5x5 into 4x4 + 4x1 + 1x4 + 1x1...
    # This is complex.
    
    # Let's try a simple heuristic: Randomly perturb and check if rank can be reduced.
    # But finding a valid lower rank decomposition is a hard problem.
    
    # Alternative: Check if the current decomposition has redundant terms.
    # If the tensor T has a lower rank, the schoolbook decomposition is not minimal.
    # We can try to compute the tensor rank via numerical methods (e.g., flattening and SVD),
    # but tensor rank is NP-hard to compute exactly.
    
    # Given the constraints, we will stick with the schoolbook decomposition unless we can quickly
    # find a valid lower rank one.
    
    # Try a few random initializations for rank 99, 98, ... down to 90
    # Use gradient descent to minimize || T - sum_r U_r (x) V_r (x) W_r ||_F
    # Then round and verify.
    
    def try_lower_rank(target_rank, max_iters=100):
        nonlocal best_rank, best_U, best_V, best_W, deadline, start_time
        if time.time() > deadline:
            return
        
        np.random.seed(42 + target_rank)
        U_curr = np.random.randn(n * m, target_rank) * 0.1
        V_curr = np.random.randn(m * p, target_rank) * 0.1
        W_curr = np.random.randn(p * n, target_rank) * 0.1
        
        # Gradient descent
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                return
            
            # Compute gradient
            # T_approx = einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            # diff = T_approx - T
            # grad_U = einsum('abc,br,cr->ar', diff, V_curr, W_curr)
            # grad_V = einsum('abc,ar,cr->br', diff, U_curr, W_curr)
            # grad_W = einsum('abc,ar,br->cr', diff, U_curr, V_curr)
            
            T_approx = np.einsum('ar,br,cr->abc', U_curr, V_curr, W_curr)
            diff = T_approx - T
            
            grad_U = np.einsum('abc,br,cr->ar', diff, V_curr, W_curr)
            grad_V = np.einsum('abc,ar,cr->br', diff, U_curr, W_curr)
            grad_W = np.einsum('abc,ar,br->cr', diff, U_curr, V_curr)
            
            U_curr -= lr * grad_U
            V_curr -= lr * grad_V
            W_curr -= lr * grad_W
            
            # Check convergence
            if np.max(np.abs(diff)) < 1e-4:
                # Round to half-integers
                U_rounded = np.round(U_curr * 2) / 2
                V_rounded = np.round(V_curr * 2) / 2
                W_rounded = np.round(W_curr * 2) / 2
                
                is_valid, err = verify(U_rounded, V_rounded, W_rounded, T)
                if is_valid:
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    return
        
        # Final check after iterations
        U_rounded = np.round(U_curr * 2) / 2
        V_rounded = np.round(V_curr * 2) / 2
        W_rounded = np.round(W_curr * 2) / 2
        is_valid, err = verify(U_rounded, V_rounded, W_rounded, T)
        if is_valid:
            best_rank = target_rank
            best_U = U_rounded
            best_V = V_rounded
            best_W = W_rounded

    # Try ranks from 99 down to 90
    for r in range(99, 89, -1):
        if time.time() > deadline:
            break
        try_lower_rank(r)
        if best_rank < r:
            break  # Found a better rank, stop
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
