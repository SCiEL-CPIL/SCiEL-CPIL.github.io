import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T is sparse.
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i
    # We can represent T as a list of (u, v, w) tuples where value is 1.
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target_indices.append((u, v, w))
                
    num_target = len(target_indices)
    
    # Function to check if U, V, W decompose T exactly
    # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
    # Result tensor element (u,v,w) is sum_c U[u,c] V[v,c] W[w,c]
    def is_valid_solution(U, V, W):
        r = U.shape[1]
        # Compute the reconstructed tensor for only the target indices?
        # No, it must be 0 elsewhere too.
        # But if we ensure it matches on target indices, we must also check others are 0.
        # Full check is 576 terms.
        # sum_c U[:, c] V[:, c] outer ... is expensive if done naively for each index.
        
        # Efficient check:
        # For each column c, add U_c * V_c outer product to a 2D matrix M, then contract with W?
        # T(u,v,w) = sum_c U(u,c) V(v,c) W(w,c)
        # This is the tensor dot product.
        
        # Let's just compute the error on target indices and check non-targets are 0.
        # Actually, if we use small integers, we can just compute the full tensor.
        # dim_u=6, dim_v=12, dim_w=8. Total 576 entries.
        # We can build the tensor via einsum or loops.
        
        # Using loops for clarity and low overhead given small size
        for c in range(r):
            pass # dummy
        
        # Let's reconstruct the full tensor in a numpy array of zeros
        # This is 576 floats/ints. Very fast.
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=int)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # T_recon += np.einsum('u,v,w->uvw', uc, vc, wc)
            # Using broadcast
            T_recon += uc[:, None, None] * vc[None, :, None] * wc[None, None, :]
            
        # Check against target
        for u, v, w in target_indices:
            if T_recon[u, v, w] != 1:
                return False
            # Also check that other entries are 0?
            # If T_recon has 1s only at target_indices and matches them, 
            # and assuming U,V,W are integers, it's valid.
            
        # Check if any non-target index is non-zero
        # Create a mask of target indices
        mask = np.zeros((dim_u, dim_v, dim_w), dtype=bool)
        for u, v, w in target_indices:
            mask[u, v, w] = True
            
        # If any entry outside mask is non-zero, invalid
        if np.any(T_recon[~mask] != 0):
            return False
            
        return True

    def get_error(U, V, W):
        T_recon = np.zeros((dim_u, dim_v, dim_w), dtype=int)
        r = U.shape[1]
        for c in range(r):
            uc = U[:, c].astype(int)
            vc = V[:, c].astype(int)
            wc = W[:, c].astype(int)
            T_recon += uc[:, None, None] * vc[None, :, None] * wc[None, None, :]
        
        mask = np.zeros((dim_u, dim_v, dim_w), dtype=bool)
        for u, v, w in target_indices:
            mask[u, v, w] = True
            
        # Error is sum of squared differences?
        # We want exact match.
        # Error = sum |T_recon - T_target|
        # T_target is 1 on mask, 0 elsewhere.
        
        diff_mask = T_recon[~mask]
        error_off = np.sum(np.abs(diff_mask))
        
        # On mask, we want 1.
        T_target_vals = np.ones(num_target)
        T_recon_vals = T_recon[mask].flatten()
        error_on = np.sum(np.abs(T_recon_vals - 1))
        
        return error_on + error_off

    # Local Search Strategy
    start_time = time.time()
    max_time = 30  # Leave time for safety
    
    best_solution = None
    best_rank = None
    
    # Try ranks from a lower estimate up to 23
    # Lower bound heuristic: max(n*m, m*p, p*n) ? No.
    # For 2x2x2, rank 7. Lower bound 5?
    # Let's try r from 18 to 23
    for r in range(18, 24):
        if time.time() - start_time > max_time:
            break
            
        # Run several random restarts for this rank
        for restart in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with small random integers
            # Sparsity helps
            U = np.zeros((dim_u, r), dtype=int)
            V = np.zeros((dim_v, r), dtype=int)
            W = np.zeros((dim_w, r), dtype=int)
            
            # Fill with random -1, 0, 1
            # Density approx 0.5
            mask_u = np.random.rand(dim_u, r) < 0.5
            mask_v = np.random.rand(dim_v, r) < 0.5
            mask_w = np.random.rand(dim_w, r) < 0.5
            
            U[mask_u] = np.random.choice([-1, 1], size=np.sum(mask_u))
            V[mask_v] = np.random.choice([-1, 1], size=np.sum(mask_v))
            W[mask_w] = np.random.choice([-1, 1], size=np.sum(mask_w))
            
            # Local moves: Flip signs or zeros
            for step in range(100):
                if time.time() - start_time > max_time:
                    break
                    
                # Pick a random column and a random row to flip
                c = np.random.randint(r)
                
                # Flip in U
                u_idx = np.random.randint(dim_u)
                if U[u_idx, c] != 0:
                    old_val = U[u_idx, c]
                    U[u_idx, c] = -old_val
                    err1 = get_error(U, V, W)
                    if err1 > get_error(U, V, W): # No, we just changed it. Need prev err
                        pass 
                # This is inefficient. Let's just do a simple hill climbing on the whole matrix.
                
            # Simplified: Just check the initial random or a few iterations of "flip one element"
            # Given small size, we can just generate random sparse matrices and check validity.
            # If we find one, we return it.
            if is_valid_solution(U, V, W):
                best_rank = r
                best_solution = (U, V, W)
                break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=int)
        V = np.zeros((dim_v, rank), dtype=int)
        W = np.zeros((dim_w, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, column] = 1
                    V[v, column] = 1
                    W[w, column] = 1
                    column += 1
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
