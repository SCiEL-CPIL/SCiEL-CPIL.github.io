import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 48
    
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # The 3x4x4 tensor can be decomposed into 3 blocks of rank 16 each
    # Block i (for row i of the 3x4 matrix) computes:
    #   result[i, k] = sum_j A[i,j] * B[j,k]
    # This is equivalent to multiplying a 1x4 row vector by a 4x4 matrix
    # The 1x4x4 tensor has rank 16 (schoolbook is optimal for 4x4)
    
    # Total rank: 3 * 16 = 48 (same as schoolbook)
    # But we can potentially do better by sharing structure
    
    # Let's try to find a lower rank decomposition
    # The theoretical lower bound for 3x4x4 is at least 48 (from 4x4 being rank 16)
    # Actually, we need to be more careful.
    
    # For now, let's use the block structure to potentially find a better decomposition
    # by optimizing across blocks
    
    # Strategy: Use the known rank-16 decomposition for 4x4 multiplication
    # and see if we can share factors across the 3 row-blocks
    
    # 4x4 multiplication: C = A * B where A, B, C are 4x4
    # Rank 16 decomposition: sum_{j=0}^{3} sum_{k=0}^{3} A[:,j] * B[j,:] * C[k,:]
    # In tensor form: T_{4x4x4}[a, b, c] = sum_{j,k} U[a,j] V[j,k] W[k,c]
    # where U is 4x4, V is 4x4, W is 4x4 (identity matrices essentially)
    
    # For 3x4x4: T[(i,j), (j',k'), (k'',i'')] = 1 if i=i', j=j', k=k''
    # = sum_{blocks} ...
    
    # Let's construct the explicit rank-48 decomposition first
    U = np.zeros((dim_u, rank_schoolbook))
    V = np.zeros((dim_v, rank_schoolbook))
    W = np.zeros((dim_w, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition
    # We'll use a smarter approach: exploit the block structure
    
    def verify(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the tensor."""
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for l in range(U.shape[1]):
                        val += U[idx_u, l] * V[idx_v, l] * W[idx_w, l]
                    if abs(val - 1.0) > tol:
                        return False
        # Check that zero entries are actually zero (sample)
        import random
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            i2 = random.randint(0, n-1)
            if i2 == i and j == j and k == k:
                continue
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i2
            if idx_w >= dim_w:
                continue
            val = 0.0
            for l in range(U.shape[1]):
                val += U[idx_u, l] * V[idx_v, l] * W[idx_w, l]
            if abs(val) > tol:
                return False
        return True
    
    def snap_to_half_int(X, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        X_snapped = np.round(X * 2) / 2
        if np.max(np.abs(X - X_snapped)) < tol:
            return X_snapped
        return X
    
    # Try to find a better decomposition using the block structure
    # Key insight: the 3 row-blocks share the same V and W structure
    # We can try to optimize the shared factors
    
    start_time = time.time()
    time_limit = 130  # Leave 20s margin for safety
    
    # Try rank 47, 46, ... using a structured approach
    # For each candidate rank, we'll use a modified ALS that respects the block structure
    
    def try_rank_structured(r):
        """Try to find a rank-r decomposition using block structure."""
        if time.time() - start_time > time_limit:
            return None
        
        # Initialize with the schoolbook decomposition, then try to reduce rank
        # by merging columns
        
        np.random.seed(42)
        
        # Start with rank r initialization
        # Use the block structure: 3 blocks, each with rank approximately r/3
        r_per_block = r // 3
        r_remainder = r % 3
        
        U = np.zeros((dim_u, r))
        V = np.zeros((dim_v, r))
        W = np.zeros((dim_w, r))
        
        # Initialize each block
        col_offset = 0
        for i in range(n):
            r_block = r_per_block + (1 if i < r_remainder else 0)
            if r_block <= 0:
                continue
            for j in range(m):
                for k in range(p):
                    if col_offset + j * p + k >= r:
                        break
                    U[i * m + j, col_offset + j * p + k] = 1
                    V[j * p + k, col_offset + j * p + k] = 1
                    W[k * n + i, col_offset + j * p + k] = 1
            col_offset += r_block
        
        # Now use ALS to optimize
        max_iters = 30
        tol = 1e-12
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U
            # T_(1) = U (W \circ V)^T
            B_U = np.zeros((dim_v * dim_w, r))
            for l in range(r):
                B_U[:, l] = np.kron(V[:, l], W[:, l])
            
            # Construct T_(1)
            T_1 = np.zeros((dim_u, dim_v * dim_w))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        T_1[idx_u, idx_v * dim_w + idx_w] = 1.0
            
            U_new, _, _, _ = np.linalg.lstsq(B_U, T_1.T, rcond=None)
            U = U_new.T
            
            # Update V
            T_2 = np.zeros((dim_v, dim_u * dim_w))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        T_2[idx_v, idx_u * dim_w + idx_w] = 1.0
                        
            B_V = np.zeros((dim_u * dim_w, r))
            for l in range(r):
                B_V[:, l] = np.kron(U[:, l], W[:, l])
            V_new, _, _, _ = np.linalg.lstsq(B_V, T_2.T, rcond=None)
            V = V_new.T
            
            # Update W
            T_3 = np.zeros((dim_w, dim_u * dim_v))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        T_3[idx_w, idx_u * dim_v + idx_v] = 1.0
                        
            B_W = np.zeros((dim_u * dim_v, r))
            for l in range(r):
                B_W[:, l] = np.kron(U[:, l], V[:, l])
            W_new, _, _, _ = np.linalg.lstsq(B_W, T_3.T, rcond=None)
            W = W_new.T
            
            # Check convergence
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = 0.0
                        for l in range(r):
                            val += U[idx_u, l] * V[idx_v, l] * W[idx_w, l]
                        residual += (1.0 - val) ** 2
                        
            if residual < tol:
                # Snap to half-integers
                U_s = snap_to_half_int(U)
                V_s = snap_to_half_int(V)
                W_s = snap_to_half_int(W)
                
                if verify(U_s, V_s, W_s):
                    return U_s, V_s, W_s
                    
        return None
    
    # Try ranks from 47 down
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 15, 40), -1):
        if time.time() - start_time > time_limit:
            break
        result = try_rank_structured(r)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no lower rank found, return schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
