"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill

    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    def _verify_fast(U, V, W):
        """Faster verification using einsum"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def _try_rank12_decomposition():
        """
        Try to find a rank-12 decomposition of the 2x3x4 matrix multiplication tensor.
        
        The tensor T[i,j,k] = 1 for all i in [0,1], j in [0,1,2], k in [0,1,2,3].
        We need to find U (6x12), V (12x12), W (8x12) such that:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        
        Strategy: Use a structured approach based on the algebraic structure.
        For rank 12, we can use a decomposition based on the fact that
        the tensor has a specific symmetry.
        
        One approach: Use the fact that for the 2x3x4 case, we can group
        the indices and use a block structure.
        """
        rank = 12
        
        # Try a specific structured decomposition
        # Based on the algebraic structure of matrix multiplication
        
        # Initialize with zeros
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strategy: Use a combination of patterns
        # For each rank dimension r, assign a pattern that covers multiple (i,j,k)
        
        # Pattern 1: Cover all (i,j,k) where i=0
        # This requires 3*4 = 12 terms, but we can share some
        
        # Let's try a different approach: use the fact that the tensor is
        # the outer product of all-ones vectors in each dimension
        # T[i,j,k] = 1 = 1*1*1
        
        # So we need a decomposition of the all-ones tensor of size 2x3x4
        # into 12 rank-1 tensors
        
        # Use a greedy approach: for each rank dimension, choose a pattern
        # that maximizes coverage
        
        # Try specific patterns based on known results
        # For 2x3x4, one known decomposition uses the following structure:
        
        # Group the rank dimensions into 4 groups of 3
        # Each group handles a subset of the (i,j,k) indices
        
        # Group 0: handle (i=0, j=0, *) and (i=1, j=0, *)
        # Group 1: handle (i=0, j=1, *) and (i=1, j=1, *)
        # Group 2: handle (i=0, j=2, *) and (i=1, j=2, *)
        # Group 3: handle remaining
        
        # Actually, let's try a more systematic approach using linear algebra
        
        # The condition is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        
        # This is a system of 24 equations in 12*6 + 12*12 + 12*8 = 72 + 144 + 96 = 312 unknowns
        # So there should be many solutions
        
        # Use a random initialization and local search
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random small values
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            # Local search: try to minimize the error
            best_error = float('inf')
            best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
            
            for step in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute current error
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                            errors.append(val - 1.0)
                error = np.sum(np.abs(errors))
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
                
                if error < 1e-10:
                    break
                
                # Try small perturbations
                improved = False
                for _ in range(20):
                    # Perturb a random entry
                    matrix = np.random.choice(['U', 'V', 'W'])
                    if matrix == 'U':
                        row = np.random.randint(n*m)
                        col = np.random.randint(rank)
                        delta = np.random.randn() * 0.1
                        old_val = U_init[row, col]
                        U_init[row, col] = old_val + delta
                        
                        # Check if improved
                        errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                                    errors.append(val - 1.0)
                        new_error = np.sum(np.abs(errors))
                        
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            U_init[row, col] = old_val
                    
                    elif matrix == 'V':
                        row = np.random.randint(m*p)
                        col = np.random.randint(rank)
                        delta = np.random.randn() * 0.1
                        old_val = V_init[row, col]
                        V_init[row, col] = old_val + delta
                        
                        errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                                    errors.append(val - 1.0)
                        new_error = np.sum(np.abs(errors))
                        
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            V_init[row, col] = old_val
                    
                    else:  # W
                        row = np.random.randint(p*n)
                        col = np.random.randint(rank)
                        delta = np.random.randn() * 0.1
                        old_val = W_init[row, col]
                        W_init[row, col] = old_val + delta
                        
                        errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                                    errors.append(val - 1.0)
                        new_error = np.sum(np.abs(errors))
                        
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            W_init[row, col] = old_val
                
                if not improved:
                    break
            
            if best_error < 1e-9:
                # Snap to nearest half-integers
                U_snapped = np.round(2 * best_U) / 2.0
                V_snapped = np.round(2 * best_V) / 2.0
                W_snapped = np.round(2 * best_W) / 2.0
                
                # Verify the snapped version
                if _verify_fast(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        return None

    def _try_rank13_decomposition():
        """Try to find a rank-13 decomposition"""
        rank = 13
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            U_init = np.random.randn(n*m, rank) * 0.5
            V_init = np.random.randn(m*p, rank) * 0.5
            W_init = np.random.randn(p*n, rank) * 0.5
            
            best_error = float('inf')
            best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
            
            for step in range(80):
                if time.time() - start_time > time_limit:
                    break
                
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                            errors.append(val - 1.0)
                error = np.sum(np.abs(errors))
                
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
                
                if error < 1e-10:
                    break
                
                improved = False
                for _ in range(20):
                    matrix = np.random.choice(['U', 'V', 'W'])
                    if matrix == 'U':
                        row = np.random.randint(n*m)
                        col = np.random.randint(rank)
                        delta = np.random.randn() * 0.1
                        old_val = U_init[row, col]
                        U_init[row, col] = old_val + delta
                        
                        errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                                    errors.append(val - 1.0)
                        new_error = np.sum(np.abs(errors))
                        
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            U_init[row, col] = old_val
                    elif matrix == 'V':
                        row = np.random.randint(m*p)
                        col = np.random.randint(rank)
                        delta = np.random.randn() * 0.1
                        old_val = V_init[row, col]
                        V_init[row, col] = old_val + delta
                        
                        errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                                    errors.append(val - 1.0)
                        new_error = np.sum(np.abs(errors))
                        
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            V_init[row, col] = old_val
                    else:
                        row = np.random.randint(p*n)
                        col = np.random.randint(rank)
                        delta = np.random.randn() * 0.1
                        old_val = W_init[row, col]
                        W_init[row, col] = old_val + delta
                        
                        errors = []
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_init[i*m+j] * V_init[j*p+k] * W_init[k*n+i])
                                    errors.append(val - 1.0)
                        new_error = np.sum(np.abs(errors))
                        
                        if new_error < error:
                            error = new_error
                            improved = True
                        else:
                            W_init[row, col] = old_val
                
                if not improved:
                    break
            
            if best_error < 1e-9:
                U_snapped = np.round(2 * best_U) / 2.0
                V_snapped = np.round(2 * best_V) / 2.0
                W_snapped = np.round(2 * best_W) / 2.0
                
                if _verify_fast(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        return None

    # Try to find lower-rank decompositions
    best_rank = rank_schoolbook
    best_result = (U_sb, V_sb, W_sb)
    
    # Try rank 12 first (the theoretical lower bound)
    result = _try_rank12_decomposition()
    if result is not None:
        best_rank = 12
        best_result = result
    else:
        # Try rank 13
        result = _try_rank13_decomposition()
        if result is not None:
            best_rank = 13
            best_result = result
    
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        U, V, W = best_result
        U = [[float(x) for x in row] for row in U]
        V = [[float(x) for x in row] for row in V]
        W = [[float(x) for x in row] for row in W]
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
