"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _make_rank17():
    """Return the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    terms = [
        ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def _compute_tensor(U, V, W, rank):
    """Compute the tensor R from factors."""
    n_a, n_b, n_c = U.shape[0], V.shape[0], W.shape[0]
    R = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return R

def _try_sat_rank16(deadline):
    """
    Try to find a rank-16 decomposition with entries in {-1, 0, 1}
    using a local search / SAT-like approach with incremental error updates.
    Returns (U, V, W) or None.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    best_solution = None
    
    for restart in range(200): # Increased restarts due to faster steps
        if time.time() > deadline - 5:
            break
        
        # Initialize: take rank-17 and zero out one column, then perturb
        U, V, W = _make_rank17()
        
        # Remove one column (randomly choose which)
        drop_col = random.randint(0, 16)
        U = np.delete(U, drop_col, axis=1)
        V = np.delete(V, drop_col, axis=1)
        W = np.delete(W, drop_col, axis=1)
        
        # Compute initial R and Error
        R = _compute_tensor(U, V, W, rank)
        err = np.sum((R - T) ** 2)
        
        if err < 1e-6:
            return U, V, W
            
        max_iters = 5000 # Increased iterations due to O(54) update cost
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Pick a random entry to flip
            which = random.randint(0, 2)  # 0=U, 1=V, 2=W
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W[c, r]
            
            # Try a different value from {-1, 0, 1}
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3 - 1  # -1->0->1->-1
                
            delta_val = new_val - old_val
            if delta_val == 0:
                continue

            # Calculate Delta R
            # If changing U[a,r], Delta R[a,b,c] = delta_val * V[b,r] * W[c,r]
            # We only need to update R[a,:,:]
            
            if which == 0:
                delta_R = delta_val * np.einsum('b,c->bc', V[:,r], W[:,r])
                # Update error for slice a
                old_slice = R[a, :, :]
                new_slice = old_slice + delta_R
                old_err_slice = np.sum((old_slice - T[a, :, :]) ** 2)
                new_err_slice = np.sum((new_slice - T[a, :, :]) ** 2)
                delta_err = new_err_slice - old_err_slice
                
                # Apply if better
                if delta_err < 0:
                    R[a, :, :] = new_slice
                    U[a, r] = new_val
                    err += delta_err
                # else revert (do nothing)
                
            elif which == 1:
                delta_R = delta_val * np.einsum('a,c->ac', U[:,r], W[:,r])
                # Update error for slice b
                old_slice = R[:, b, :]
                new_slice = old_slice + delta_R
                old_err_slice = np.sum((old_slice - T[:, b, :]) ** 2)
                new_err_slice = np.sum((new_slice - T[:, b, :]) ** 2)
                delta_err = new_err_slice - old_err_slice
                
                if delta_err < 0:
                    R[:, b, :] = new_slice
                    V[b, r] = new_val
                    err += delta_err
                    
            else:
                delta_R = delta_val * np.einsum('a,b->ab', U[:,r], V[:,r])
                # Update error for slice c
                old_slice = R[:, :, c]
                new_slice = old_slice + delta_R
                old_err_slice = np.sum((old_slice - T[:, :, c]) ** 2)
                new_err_slice = np.sum((new_slice - T[:, :, c]) ** 2)
                delta_err = new_err_slice - old_err_slice
                
                if delta_err < 0:
                    R[:, :, c] = new_slice
                    W[c, r] = new_val
                    err += delta_err

            if err < 1e-6:
                return U, V, W
        
        # Check final
        if err < 1e-6:
            return U, V, W
    
    return None

def _try_structural_rank16(deadline):
    """
    Try structural modifications to the rank-17 solution to get rank 16.
    Uses incremental error updates for speed.
    """
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    rank = 16
    
    random.seed(42)
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Start from rank-17, drop a random column
        U, V, W = _make_rank17()
        drop_col = random.randint(0, 16)
        U = np.delete(U, drop_col, axis=1)
        V = np.delete(V, drop_col, axis=1)
        W = np.delete(W, drop_col, axis=1)
        
        # Compute initial R and Error
        R = _compute_tensor(U, V, W, rank)
        err = np.sum((R - T) ** 2)
        
        if err < 1e-6:
            return U, V, W
            
        current_err = err
        for it in range(5000):
            if time.time() > deadline - 3:
                break
            
            # Pick random entry
            which = random.randint(0, 2)
            if which == 0:
                a, r = random.randint(0, n_a-1), random.randint(0, rank-1)
                old_val = U[a, r]
            elif which == 1:
                b, r = random.randint(0, n_b-1), random.randint(0, rank-1)
                old_val = V[b, r]
            else:
                c, r = random.randint(0, n_c-1), random.randint(0, rank-1)
                old_val = W[c, r]
            
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                new_val = (new_val + 1) % 3 - 1
                
            delta_val = new_val - old_val
            if delta_val == 0:
                continue

            # Calculate Delta R and Delta Error
            if which == 0:
                delta_R = delta_val * np.einsum('b,c->bc', V[:,r], W[:,r])
                old_slice = R[a, :, :]
                new_slice = old_slice + delta_R
                old_err_slice = np.sum((old_slice - T[a, :, :]) ** 2)
                new_err_slice = np.sum((new_slice - T[a, :, :]) ** 2)
                delta_err = new_err_slice - old_err_slice
                new_err = current_err + delta_err
                
                # Accept if better, or with small probability if worse (annealing)
                if new_err <= current_err or (it > 1000 and random.random() < 0.01):
                    R[a, :, :] = new_slice
                    U[a, r] = new_val
                    current_err = new_err
                # else revert
                
            elif which == 1:
                delta_R = delta_val * np.einsum('a,c->ac', U[:,r], W[:,r])
                old_slice = R[:, b, :]
                new_slice = old_slice + delta_R
                old_err_slice = np.sum((old_slice - T[:, b, :]) ** 2)
                new_err_slice = np.sum((new_slice - T[:, b, :]) ** 2)
                delta_err = new_err_slice - old_err_slice
                new_err = current_err + delta_err
                
                if new_err <= current_err or (it > 1000 and random.random() < 0.01):
                    R[:, b, :] = new_slice
                    V[b, r] = new_val
                    current_err = new_err
                    
            else:
                delta_R = delta_val * np.einsum('a,b->ab', U[:,r], V[:,r])
                old_slice = R[:, :, c]
                new_slice = old_slice + delta_R
                old_err_slice = np.sum((old_slice - T[:, :, c]) ** 2)
                new_err_slice = np.sum((new_slice - T[:, :, c]) ** 2)
                delta_err = new_err_slice - old_err_slice
                new_err = current_err + delta_err
                
                if new_err <= current_err or (it > 1000 and random.random() < 0.01):
                    R[:, :, c] = new_slice
                    W[c, r] = new_val
                    current_err = new_err

            if current_err < 1e-6:
                return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    n_a, n_b, n_c = n*m, m*p, p*n
    
    # First, try to find rank 16
    T = _build_tensor(n, m, p)
    
    # Try SAT-style search for rank 16
    result_16 = _try_sat_rank16(deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try structural approach
    result_16b = _try_structural_rank16(deadline)
    if result_16b is not None:
        U, V, W = result_16b
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 17
    U, V, W = _make_rank17()
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    rank_fb = n * m * p
    U_fb = np.zeros((n_a, rank_fb))
    V_fb = np.zeros((n_b, rank_fb))
    W_fb = np.zeros((n_c, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
