<<<<<<< SEARCH
def _try_merge_strassen_terms(U55, V55, W55, n, m, p, target_rank, deadline):
    """
    Try to merge pairs of Strassen terms from different B-blocks.
    
    The idea: if two rank-1 terms have similar U or V patterns, they might
    be combinable into a single term with adjusted W.
    
    We try all pairs of terms and check if their difference can be expressed
    as a single rank-1 term.
    """
    rank55 = 55
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    # For each pair of terms, try to see if we can eliminate one by adjusting others
    # Strategy: pick a subset of terms to keep, then solve for adjusted coefficients
    
    # First, try dropping one term at a time and see if the remaining can be adjusted
    for drop_idx in range(rank55):
        if time.time() > deadline:
            return None
        
        # Create a matrix where each column is one of the 55 terms (except drop_idx)
        keep_idx = [i for i in range(rank55) if i != drop_idx]
        U_keep = U55[:, keep_idx]  # shape (n*m, 54)
        V_keep = V55[:, keep_idx]  # shape (m*p, 54)
        W_keep = W55[:, keep_idx]  # shape (p*n, 54)
        
        # The tensor to reconstruct: sum of all 55 terms minus the dropped one
        # T_target = T_all - T_drop
        # We want: sum_{r in keep} c_r * (U_keep[:,r] ⊗ V_keep[:,r] ⊗ W_keep[:,r]) = T_target
        
        # This is a linear system in the coefficients c_r
        # Flatten the tensor equation: for each (a,b,c) index:
        # sum_r c_r * U_keep[a,r] * V_keep[b,r] * W_keep[c,r] = T_target[a,b,c]
        
        # This is a 3D tensor equation. We can write it as:
        # For each (a,b): sum_r c_r * U_keep[a,r] * V_keep[b,r] * W_keep[c,r] = T_target[a,b,c]
        # This is linear in c_r for each (a,b,c).
        
        # Build the linear system: A @ c = b
        # A[i, r] = U_keep[a,r] * V_keep[b,r] * W_keep[c,r] for the i-th equation
        # But we have (n*m)*(m*p)*(p*n) equations, which is too many.
        
        # Instead, use a least-squares approach with random sampling
        # Sample a subset of (a,b,c) indices to form the linear system
        
        n_eq = min(500, (n*m) * (m*p) * (p*n))
        random.seed(42)
        a_indices = [random.randint(0, n*m-1) for _ in range(n_eq)]
        b_indices = [random.randint(0, m*p-1) for _ in range(n_eq)]
        c_indices = [random.randint(0, p*n-1) for _ in range(n_eq)]
        
        # Build A matrix: n_eq x 54
        A = np.zeros((n_eq, 54), dtype=np.float64)
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            for r in range(54):
                A[eq, r] = U_keep[a, r] * V_keep[b, r] * W_keep[c, r]
        
        # Build b vector: T_target at sampled indices
        # T_target = sum of all 55 terms - dropped term
        # = T_all - T_drop
        # T_all[a,b,c] = 1 if (a,b,c) is a valid (i,j,k) mapping, else 0
        # T_drop[a,b,c] = U55[a,drop_idx] * V55[b,drop_idx] * W55[c,drop_idx]
        
        b_vec = np.zeros(n_eq, dtype=np.float64)
        for eq in range(n_eq):
            a, b, c = a_indices[eq], b_indices[eq], c_indices[eq]
            # Check if (a,b,c) is a valid multiplication tensor entry
            # a = i*m+j, b = j*p+k, c = k*n+i
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                b_vec[eq] = 1.0
            # Subtract the dropped term
            b_vec[eq] -= U55[a, drop_idx] * V55[b, drop_idx] * W55[c, drop_idx]
        
        # Solve least squares: A @ c = b_vec
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
        except Exception:
            continue
        
        # Check if the solution is valid: residuals should be small
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        # Check if coefficients are close to integers or half-integers
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        # Construct the new decomposition
        U_new = np.zeros((n*m, 54), dtype=np.float64)
        V_new = np.zeros((m*p, 54), dtype=np.float64)
        W_new = np.zeros((p*n, 54), dtype=np.float64)
        for r in range(54):
            U_new[:, r] = coeffs[r] * U_keep[:, r]
            V_new[:, r] = coeffs[r] * V_keep[:, r]
            W_new[:, r] = coeffs[r] * W_keep[:, r]
        
        # Verify
        if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None
=======
def _try_merge_strassen_terms(U55, V55, W55, n, m, p, target_rank, deadline):
    """
    Try to reduce rank by projecting the rank-55 solution to lower rank
    using iterative hard thresholding / ALS approach.
    """
    rank55 = 55
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    # Build the full multiplication tensor for reference
    T_full = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0
    
    # Try to find a rank-r decomposition by:
    # 1. Start with random rank-r factors
    # 2. Iteratively update each factor to best match T_full
    # 3. Check for exactness
    
    for r in range(target_rank, min(target_rank + 5, 55)):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random projection of the 55-term solution
            # Pick r random linear combinations of the 55 terms
            U_r = np.zeros((n*m, r), dtype=np.float64)
            V_r = np.zeros((m*p, r), dtype=np.float64)
            W_r = np.zeros((p*n, r), dtype=np.float64)
            
            for col in range(r):
                # Random combination of 2-4 of the 55 terms
                n_terms = random.randint(2, 4)
                idx = random.sample(range(55), n_terms)
                coeffs = [random.choice([1, -1]) for _ in range(n_terms)]
                for c, i in zip(coeffs, idx):
                    U_r[:, col] += c * U55[:, i]
                    V_r[:, col] += c * V55[:, i]
                    W_r[:, col] += c * W55[:, i]
            
            # ALS iterations
            for it in range(20):
                if time.time() > deadline:
                    break
                
                # Update U: for each column a of U, solve least squares
                # T[a,:,:] ≈ sum_r U[a,r] * V[:,r] ⊗ W[:,r]
                # Reshape: for each a, T[a] is (m*p, p*n) matrix
                # T[a] ≈ U[a,:] @ (V^T ⊗ W^T) ... this is complex
                
                # Simpler: update U by solving for each a
                # For fixed V, W: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each a: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                # So: T[a].reshape(-1) = U[a,:] @ M_a where M_a[r, :] = (V[:,r] ⊗ W[:,r]).reshape(-1)
                
                # Build M matrix: r x (m*p * p*n)
                # M[r, b*p*n + c] = V[b,r] * W[c,r]
                M = np.zeros((r, m*p * p*n), dtype=np.float64)
                for rr in range(r):
                    for b in range(m*p):
                        for c in range(p*n):
                            M[rr, b * p*n + c] = V_r[b, rr] * W_r[c, rr]
                
                # Update U
                for a in range(n*m):
                    target = T_full[a].reshape(-1)
                    # Solve M.T @ U[a,:] = target
                    try:
                        U_r[a, :], res, _, _ = np.linalg.lstsq(M.T, target, rcond=None)
                    except Exception:
                        break
                
                # Update V similarly
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each b: T[:,b,:] = sum_r V[b,r] * (U[:,r] ⊗ W[:,r])
                # M_V[r, a*p*n + c] = U[a,r] * W[c,r]
                M_V = np.zeros((r, n*m * p*n), dtype=np.float64)
                for rr in range(r):
                    for a in range(n*m):
                        for c in range(p*n):
                            M_V[rr, a * p*n + c] = U_r[a, rr] * W_r[c, rr]
                
                for b in range(m*p):
                    target = T_full[:, b, :].reshape(-1)
                    try:
                        V_r[b, :], res, _, _ = np.linalg.lstsq(M_V.T, target, rcond=None)
                    except Exception:
                        break
                
                # Update W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each c: T[:,:,c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
                # M_W[r, a*m*p + b] = U[a,r] * V[b,r]
                M_W = np.zeros((r, n*m * m*p), dtype=np.float64)
                for rr in range(r):
                    for a in range(n*m):
                        for b in range(m*p):
                            M_W[rr, a * m*p + b] = U_r[a, rr] * V_r[b, rr]
                
                for c in range(p*n):
                    target = T_full[:, :, c].reshape(-1)
                    try:
                        W_r[c, :], res, _, _ = np.linalg.lstsq(M_W.T, target, rcond=None)
                    except Exception:
                        break
                
                # Check exactness
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    if _is_valid_entries(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                        return U_r.tolist(), V_r.tolist(), W_r.tolist()
            
            # If we got here, this restart didn't find a solution
            pass
    
    return None
>>>>>>> REPLACE

