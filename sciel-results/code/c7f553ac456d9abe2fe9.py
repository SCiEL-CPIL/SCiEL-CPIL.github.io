import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the Schoolbook Solution (Rank 63)
    # This is guaranteed to be correct and scores 1.0.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Precompute target tensor indices for fast verification
    # T[a, b, c] = 1 if (a,b,c) corresponds to (i,j,k)
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    # Total size of tensor space
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    total_entries = dim_U * dim_V * dim_W
    
    def verify(U, V, W):
        """
        Verify if U, V, W (floats, expected to be half-integers) 
        exactly decompose the tensor.
        Returns True if valid.
        """
        # Check if they are close to half-integers
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        
        if not np.allclose(U * 2, U_int, atol=1e-9):
            return False
        if not np.allclose(V * 2, V_int, atol=1e-9):
            return False
        if not np.allclose(W * 2, W_int, atol=1e-9):
            return False
            
        U_int = U_int.astype(np.int64)
        V_int = V_int.astype(np.int64)
        W_int = W_int.astype(np.int64)
        
        r = U.shape[1]
        
        # Check non-zero entries (must sum to 8)
        for a, b, c in target_indices:
            s = 0
            for k in range(r):
                s += U_int[a, k] * V_int[b, k] * W_int[c, k]
            if s != 8:
                return False
                
        # Check zero entries (must sum to 0)
        # To be fast, we can check a subset or use the fact that 
        # if the rank is low, the support is small.
        # For rank < 63, the zero check is critical.
        # We iterate over all a, b, c not in target_indices.
        # Since total_entries is ~4000, and r is ~60, this is 240k ops. Fast.
        
        # Create a set of target indices for O(1) lookup
        target_set = set(target_indices)
        
        # Optimization: Instead of checking all 4000, we can check 
        # random samples if time is tight, but 4000 is small enough.
        for a in range(dim_U):
            for b in range(dim_V):
                for c in range(dim_W):
                    if (a, b, c) in target_set:
                        continue
                    s = 0
                    for k in range(r):
                        s += U_int[a, k] * V_int[b, k] * W_int[c, k]
                    if s != 0:
                        return False
                        
        return True

    def search_rank(r, time_limit):
        """
        Attempt to find a rank-r decomposition using ALS.
        """
        np.random.seed(42) # Reproducibility for debugging, but we want diversity? 
        # Let's use random seed each call
        
        # Initialize with random Gaussian
        U = np.random.randn(dim_U, r)
        V = np.random.randn(dim_V, r)
        W = np.random.randn(dim_W, r)
        
        # Precompute target tensor T as a sparse representation for ALS
        # T is zero except at target_indices where it is 1.
        
        for it in range(50): # Max 50 iterations
            if time.time() > time_limit:
                break
                
            # Update U
            # For each a, solve min || T[a,:,:] - U[a,:] @ M_a ||
            # where M_a[b, c, k] is not a matrix, it's a tensor contraction.
            # T[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
            # For fixed a, let u_a = U[a, :].
            # We want u_a such that for all b, c:
            # sum_k u_a[k] (V[b, k] W[c, k]) = T[a, b, c]
            # Let M_a be a matrix of size (dim_V * dim_W) x r.
            # M_a[b * dim_W + c, k] = V[b, k] * W[c, k]
            # Then T_a (flattened) = M_a @ u_a
            
            # Construct M_a efficiently?
            # V is (dim_V, r), W is (dim_W, r).
            # M_a is the "outer product" of V and W in the k-dimension, flattened in b,c.
            
            for a in range(dim_U):
                # Build M_a
                # M_a is (dim_V * dim_W) x r
                # We can do this by broadcasting
                # V[:, np.newaxis, :] * W[np.newaxis, :, :] -> (dim_V, dim_W, r)
                # Reshape to (dim_V * dim_W, r)
                M_a = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, r)
                
                # Target vector for a
                # T[a, :, :] is 1 at specific (b,c) pairs, 0 elsewhere.
                # We need to build t_a of size dim_V * dim_W
                t_a = np.zeros(dim_V * dim_W)
                # Fill t_a based on target_indices for this a
                for b, c in target_indices:
                    if b[0] == a: # target_indices stores (a,b,c)
                        t_a[b[1] * dim_W + b[2]] = 1.0
                        
                # Solve least squares
                # M_a is (N, r). t_a is (N,).
                # u_a = (M_a^T M_a)^-1 M_a^T t_a
                try:
                    U[a, :], residuals, rank, sv = np.linalg.lstsq(M_a, t_a, rcond=None)
                except:
                    pass
                    
            # Update V
            for b in range(dim_V):
                M_b = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, r)
                t_b = np.zeros(dim_U * dim_W)
                for a, c in target_indices:
                    if a[1] == b:
                        t_b[a[0] * dim_W + a[2]] = 1.0
                try:
                    V[b, :], residuals, rank, sv = np.linalg.lstsq(M_b, t_b, rcond=None)
                except:
                    pass

            # Update W
            for c in range(dim_W):
                M_c = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, r)
                t_c = np.zeros(dim_U * dim_V)
                for a, b in target_indices:
                    if a[2] == c:
                        t_c[a[0] * dim_V + a[1]] = 1.0
                try:
                    W[c, :], residuals, rank, sv = np.linalg.lstsq(M_c, t_c, rcond=None)
                except:
                    pass
            
            # Check for convergence/validity periodically
            if it % 5 == 0:
                # Round and verify
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                if verify(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded

        return None

    # Attempt to find lower rank
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # Try ranks 62 down to 50
    # We only try a few initializations per rank to stay within time
    for r in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try 3 random initializations
        for init in range(3):
            if time.time() > deadline:
                break
            res = search_rank(r, deadline - 5)
            if res is not None:
                U, V, W = res
                best_U = U
                best_V = V
                best_W = W
                best_rank = r
                # If we found a lower rank, we can stop or continue?
                # Continue to see if we can go even lower, but time is tight.
                # Let's break and return the improvement.
                break

    # Return the best found
    # Convert to list of lists as per interface requirement
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
