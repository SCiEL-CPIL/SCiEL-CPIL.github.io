"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    We use a structured search approach:
    1. Start with the schoolbook rank 54 decomposition
    2. Try to find lower rank decompositions using algebraic constraints
    3. The key insight is that for small matrices, we can search over possible 
       rank-r decompositions by solving systems of polynomial equations
    
    For 3x3 * 3x6, the theoretical lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18,
    but the actual minimum is known to be higher. We'll search systematically.
    
    Strategy: Use the fact that the multiplication tensor T has components T[i,j,k] = 1 
    if the output index (k*n+i) depends on input indices (i*m+j) and (j*p+k).
    
    For a rank-r decomposition, we need vectors u_c, v_c, w_c such that:
    T = sum_c u_c ⊗ v_c ⊗ w_c
    
    This means: for all (i,j,k) where the output element (C)_{i,k} = sum_j A_{i,j} * B_{j,k},
    we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{correct}
    
    where delta_{correct} = 1 if (i,j,k) corresponds to a valid multiplication term, else 0.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Verify all required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This term should contribute 1 to C[i,k]
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        # Also check that no spurious contributions exist
        # For each output (i,k), only terms with the correct j should contribute
        # This is automatically satisfied if the above holds for all (i,j,k)
        # and the structure is correct. 
        # However, we must also ensure that the tensor is zero elsewhere.
        # The tensor T is non-zero only at (i*m+j, j*p+k, k*n+i).
        # Our check above verifies T[valid] == 1.
        # We should also verify that T[invalid] == 0.
        # But the tensor is sparse. The "invalid" entries are all others.
        # Due to symmetry and structure, if the valid ones are 1, are the others 0?
        # Not necessarily. We must check a sample of invalid entries or all.
        # Given small sizes, we can check all.
        
        # Build the full tensor check
        # T[a, b, c] = sum_c U[a, c] V[b, c] W[c, c] (index collision in variable name c, using r)
        # Let's just iterate over all possible a, b, c indices.
        # a in [0, n*m), b in [0, m*p), c_idx in [0, p*n)
        
        # Map from (i, j, k) to (a, b, c_idx)
        # a = i*m + j
        # b = j*p + k
        # c_idx = k*n + i
        
        # Create a set of valid (a, b, c_idx) tuples
        valid_indices = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    valid_indices.add((a, b, c_idx))
        
        # Check all possible (a, b, c_idx)
        for a in range(n * m):
            for b in range(m * p):
                for c_idx in range(p * n):
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c_idx][r]
                    
                    if (a, b, c_idx) in valid_indices:
                        if abs(s - 1) > 1e-9:
                            return False
                    else:
                        if abs(s) > 1e-9:
                            return False
                            
        return True

    def try_rank_r_decomposition(r, max_iter=5000, timeout=60, seed=None):
        """
        Try to find a rank-r decomposition using random initialization and gradient descent
        on the constraint that the tensor is exactly decomposed.
        
        We formulate this as: minimize ||T - sum_c u_c ⊗ v_c ⊗ w_c||^2
        subject to the constraint that the solution is exact (residual = 0).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        start_time = time.time()
        
        # Pre-compute indices for vectorization
        # We need to compute sum over r of U[a,r] V[b,r] W[c,r] for all a,b,c
        # This is equivalent to tensor contraction.
        
        # Initialize with random small values
        U = np.random.uniform(-1, 1, (n * m, r))
        V = np.random.uniform(-1, 1, (m * p, r))
        W = np.random.uniform(-1, 1, (p * n, r))
        
        # Target tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        def compute_residual_and_grad(U, V, W):
            """
            Compute residual and gradients using NumPy.
            Residual R = T - sum_r U[:,r] V[:,r] W[:,r] (broadcasting)
            Actually, the term is E[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            This is an Einstein summation: np.einsum('ar,br,cr->abc', U, V, W)
            """
            E = np.einsum('ar,br,cr->abc', U, V, W)
            R = E - T
            res = np.sum(R**2)
            
            # Gradients
            # d/dU[a,r] sum R^2 = 2 * sum_{b,c} R[a,b,c] V[b,r] W[c,r]
            # This is 2 * np.einsum('abc,br,cr->ar', R, V, W)
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', R, U, V)
            
            return res, grad_U, grad_V, grad_W

        lr = 0.01
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                break
            
            res, grad_U, grad_V, grad_W = compute_residual_and_grad(U, V, W)
            
            if res < best_res:
                best_res = res
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if res < 1e-10:
                # Check exactness
                U_list = best_U.tolist()
                V_list = best_V.tolist()
                W_list = best_W.tolist()
                
                # Try to round to half-integers
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if validate(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                
                # If rounding fails, try raw values (though task requires int/half-int)
                # The task says "integer/half-integer entries". 
                # If we can't round, we might need to continue or fail.
                # Let's check if raw values are close enough to be considered valid 
                # if we snap them? No, the evaluator checks exactness.
                # If we found a solution with res < 1e-10, it's effectively exact.
                # But we must return int/half-int.
                # If rounding breaks it, we might need to solve a least squares 
                # to find the closest half-integer solution?
                # For now, if rounding fails, we continue searching.
                
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.999
            
        # After loop, check best
        if best_U is not None and best_res < 1e-10:
             U_list = best_U.tolist()
             V_list = best_V.tolist()
             W_list = best_W.tolist()
             
             U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
             V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
             W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
             
             if validate(U_rounded, V_rounded, W_rounded):
                 return {"U": U_rounded, "V": V_rounded, "W": W_rounded}

        return None
    
    # First, try to find a decomposition with rank < 54
    # The minimum rank for 3x3 * 3x6 is not well-known, but let's search
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, n*p) = max(9, 18, 18) = 18
    # But this is a weak bound. Let's try from 24 upwards (heuristic)
    
    best = None
    start_time_global = time.time()
    
    # We have 120s total. Reserve 10s for overhead/return. 110s for search.
    # We'll iterate through ranks.
    # For each rank, we have limited time.
    
    ranks_to_try = list(range(18, 54))
    random.shuffle(ranks_to_try) # Explore different ranks
    
    for r in ranks_to_try:
        if time.time() - start_time_global > 110:
            break
            
        # Allocate more time to lower ranks as they are more likely to be optimal?
        # Or just fixed time per rank?
        # Let's give 20s per rank attempt, with multiple restarts if fast.
        
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() - start_time_global > 110:
                break
                
            result = try_rank_r_decomposition(r, max_iter=2000, timeout=15, seed=restart)
            if result is not None:
                best = result
                found = True
                break
        
        if found:
            break
            
    if best is not None:
        return best
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
