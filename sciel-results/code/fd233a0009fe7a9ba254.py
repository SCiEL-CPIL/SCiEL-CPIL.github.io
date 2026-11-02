import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _check_rank1_residual(R, tol=1e-9):
    """
    Check if tensor R (shape a, b, c) is rank 1.
    Returns (success, u, v, w) where R ~= outer(u, outer(v, w))
    """
    a, b, c = R.shape
    # Find a non-zero pivot
    pivot_found = False
    a0, b0, c0 = -1, -1, -1
    for i in range(a):
        for j in range(b):
            for k in range(c):
                if abs(R[i, j, k]) > tol:
                    a0, b0, c0 = i, j, k
                    pivot_found = True
                    break
            if pivot_found: break
        if pivot_found: break
    
    if not pivot_found:
        return True, np.zeros(a), np.zeros(b), np.zeros(c)
    
    val = R[a0, b0, c0]
    
    # Assume w_c0 is the scaling factor
    # w_c = R[a0, b0, c] / val
    w = R[a0, b0, :] / val
    # v_b = R[a0, b, c0] / val
    v = R[a0, :, c0] / val
    # u_a = R[a, b0, c0] / val
    u = R[:, b0, c0] / val
    
    # Verify
    # R_recon = np.einsum('a,b,c->abc', u, v, w)
    # Check max error
    err = 0.0
    # Sample check for speed, or full check if small
    for i in range(a):
        for j in range(b):
            for k in range(c):
                recon = u[i] * v[j] * w[k]
                if abs(recon - R[i, j, k]) > tol:
                    return False, None, None, None
    return True, u, v, w

def _try_reduce_rank(U, V, W, n, m, p, deadline, seed=42):
    """
    Attempt to reduce rank by finding linear dependencies.
    """
    rng = np.random.RandomState(seed)
    R = U.shape[1]
    a_dim, b_dim, c_dim = U.shape[0], V.shape[0], W.shape[0]
    
    # Pre-compute non-zero indices of the target tensor T
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
    
    # Also need to check zero entries. 
    # For speed, we only check a subset of zero entries in the residual calculation?
    # No, the residual MUST be zero everywhere T is zero.
    # So we must check all (a,b,c) not in target_indices.
    # Total entries: 9 * 18 * 18 = 2916. Target: 54. Zero: 2862.
    # This is small enough to check fully in numpy.
    
    all_indices = np.array([(a, b, c) for a in range(a_dim) for b in range(b_dim) for c in range(c_dim)])
    target_set = set(target_indices)
    zero_indices = [idx for idx in all_indices if tuple(idx) not in target_set]
    
    # Convert to arrays for fast indexing
    a_zero = np.array([z[0] for z in zero_indices])
    b_zero = np.array([z[1] for z in zero_indices])
    c_zero = np.array([z[2] for z in zero_indices])
    
    a_target = np.array([t[0] for t in target_indices])
    b_target = np.array([t[1] for t in target_indices])
    c_target = np.array([t[2] for t in target_indices])
    
    # Current effective rank
    active_cols = list(range(R))
    
    for step in range(10000):
        if time.time() > deadline:
            break
        
        if not active_cols:
            break
            
        # Pick a random active column to try to zero out
        r_idx = active_cols[rng.randint(0, len(active_cols)-1)]
        
        # Calculate Residual R for column r_idx
        # R[a,b,c] = T[a,b,c] - sum_{r != r_idx} U[a,r]V[b,r]W[c,r]
        
        # Efficient calculation:
        # Sum all columns, then subtract column r_idx
        # Sum = einsum('ar,br,cr->abc', U, V, W)
        # This is expensive if done from scratch every time.
        # But dimensions are small.
        
        # Let's just compute the sum of all other columns.
        # U_other = np.delete(U, r_idx, axis=1)
        # This copies memory.
        
        # Better: Compute total sum, subtract col r_idx contribution.
        # Total[a,b,c] = sum_r U[a,r]V[b,r]W[c,r]
        # Col_r[a,b,c] = U[a,r_idx]V[b,r_idx]W[c,r_idx]
        
        # We want Residual = T - (Total - Col_r) = T - Total + Col_r
        # Wait. The equation is:
        # sum_{r in active} U[a,r]V[b,r]W[c,r] = T[a,b,c]
        # If we fix all r != r_idx, let S = sum_{r != r_idx} ...
        # Then we need U[a,r_idx]V[b,r_idx]W[c,r_idx] = T[a,b,c] - S
        # So Residual R = T - S.
        
        # Calculate S (sum of all other columns)
        # We can compute Total = np.einsum('ar,br,cr->abc', U, V, W)
        # Then S = Total - np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        Total = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        Col_r = np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx], optimize=True)
        S = Total - Col_r
        
        T_tensor = np.zeros_like(Total)
        T_tensor[a_target, b_target, c_target] = 1.0
        
        Residual = T_tensor - S
        
        # Check if Residual is rank 1
        success, u, v, w = _check_rank1_residual(Residual, tol=1e-8)
        
        if success:
            # Update column r_idx
            U[:, r_idx] = u
            V[:, r_idx] = v
            W[:, r_idx] = w
            
            # Check if this column is now zero
            if np.allclose(U[:, r_idx], 0, atol=1e-9) and np.allclose(V[:, r_idx], 0, atol=1e-9) and np.allclose(W[:, r_idx], 0, atol=1e-9):
                active_cols.remove(r_idx)
                
            # Optional: Perturb other columns slightly to escape local minima?
            # For now, just keep the solution.
            
        else:
            # Perturb the current column slightly to change the residual
            # This helps in finding a different residual that might be rank 1
            scale = 0.1
            U[:, r_idx] += rng.randn(a_dim) * scale
            V[:, r_idx] += rng.randn(b_dim) * scale
            W[:, r_idx] += rng.randn(c_dim) * scale
            
            # Also perturb a random other column slightly to break symmetry
            if len(active_cols) > 1:
                other_r = active_cols[rng.randint(0, len(active_cols)-1)]
                if other_r != r_idx:
                    U[:, other_r] += rng.randn(a_dim) * scale * 0.1
                    V[:, other_r] += rng.randn(b_dim) * scale * 0.1
                    W[:, other_r] += rng.randn(c_dim) * scale * 0.1

    # Final Verification
    if not active_cols:
        return U, V, W, 0
        
    # Construct final matrices with only active columns
    U_final = U[:, active_cols]
    V_final = V[:, active_cols]
    W_final = W[:, active_cols]
    
    # Verify
    Total = np.einsum('ar,br,cr->abc', U_final, V_final, W_final, optimize=True)
    T_tensor = np.zeros_like(Total)
    T_tensor[a_target, b_target, c_target] = 1.0
    
    if np.allclose(Total, T_tensor, atol=1e-8):
        return U_final, V_final, W_final, len(active_cols)
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # Try to reduce rank
    # We try multiple seeds/restarts
    for seed in [42, 123, 999]:
        if time.time() > deadline - 10:
            break
            
        U_copy = U_school.copy()
        V_copy = V_school.copy()
        W_copy = W_school.copy()
        
        # Add small noise to break symmetry
        noise_scale = 0.05
        U_copy += np.random.RandomState(seed).randn(*U_copy.shape) * noise_scale
        V_copy += np.random.RandomState(seed + 1).randn(*V_copy.shape) * noise_scale
        W_copy += np.random.RandomState(seed + 2).randn(*W_copy.shape) * noise_scale
        
        U_res, V_res, W_res, rank_res = _try_reduce_rank(U_copy, V_copy, W_copy, n, m, p, deadline, seed=seed)
        
        if U_res is not None and rank_res < best_rank:
            best_U, best_V, best_W = U_res, V_res, W_res
            best_rank = rank_res
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
