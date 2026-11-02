<<<<<<< SEARCH
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks as the primary approach.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Try Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_2x2x4(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    # Final verification
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
def _try_rank_reduction(U, V, W, T, time_limit=None):
    """
    Attempt to reduce the rank of the decomposition by identifying linearly dependent
    rank-1 terms. We flatten the tensor and use Gaussian elimination to find a minimal
    set of linearly independent terms that still span the tensor.
    
    The key insight: if we have rank R terms, we can check if some subset of size R' < R
    still spans the same tensor by examining the 3D structure.
    """
    if time_limit is None:
        time_limit = 30.0
    
    start = time.time()
    R = U.shape[1]
    U_dim, V_dim, W_dim = U.shape[0], V.shape[0], W.shape[0]
    
    # Build the full tensor from current decomposition
    T_current = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for r in range(R):
        T_current += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Verify it matches target
    if not np.allclose(T_current, T, atol=1e-9):
        return U, V, W, R
    
    # Try to find linear dependencies among the rank-1 terms
    # Each rank-1 term u_r * v_r * w_r can be viewed as a vector in R^(U_dim*V_dim*W_dim)
    # But more efficiently, we can use the structure: for each pair (i,j) in U and V,
    # the contribution to W is u[i,r]*v[j,r]*w[:,r]
    
    # Alternative approach: use the fact that the tensor is in a 3D space.
    # We can try removing terms one at a time and checking if the remaining terms
    # can still reproduce T (allowing recombination).
    
    # More practical: try subsets of terms. Start by checking if we can drop any single term.
    # If dropping term r still allows exact reproduction with the remaining R-1 terms, great.
    
    # Actually, a better approach: the 42 Strassen terms may have linear dependencies
    # across different blocks. Let's check the rank of the coefficient structure.
    
    # For each output position (i,k) in C (2x6 = 12 positions), the contribution
    # from each term r is sum over j of u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r]
    # Wait, that's not quite right. Let me think again.
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    # Our decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[...]
    
    # For a fixed (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # if (i,j,k) is a valid index, else 0.
    
    # The number of constraints is 2*4*6 = 48 (one for each (i,j,k) tuple).
    # We have 42 terms, each with 8+24+12 = 44 coefficients, but they're coupled.
    
    # Let's try a different approach: since we know the structure, let's see if
    # we can merge or eliminate terms by exploiting the specific pattern.
    
    # For now, let's try a simple approach: check if any single term can be removed
    # by solving for the remaining terms. This is expensive, so we'll be selective.
    
    # Actually, let's try a smarter approach: the 6 blocks of 7 Strassen terms each.
    # Within each block, the 7 terms are linearly independent (that's the Strassen result).
    # But across blocks, there might be dependencies.
    
    # Let's build a matrix where each column corresponds to a term, and each row
    # corresponds to a (i,j,k) constraint. The entry is the contribution of that
    # term to that constraint.
    
    num_constraints = n * m * p  # 48
    constraint_matrix = np.zeros((num_constraints, R), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c_idx = i * m * p + j * p + k
                for r in range(R):
                    constraint_matrix[c_idx, r] = U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    # Now we want to find the minimum number of columns that can still represent
    # the target vector t (where t[c_idx] = 1 for all valid (i,j,k))
    target_vec = np.ones(num_constraints, dtype=np.float64)
    
    # Check: does constraint_matrix @ ones(R) = target_vec?
    # Actually, our decomposition should satisfy this. Let's verify.
    current_sum = constraint_matrix @ np.ones(R, dtype=np.float64)
    if not np.allclose(current_sum, target_vec, atol=1e-9):
        return U, V, W, R
    
    # Now, we want to find a sparse representation: find x with minimum support
    # such that constraint_matrix @ x = target_vec.
    
    # This is a combinatorial problem. Let's try greedy: start with all terms,
    # and try to remove one at a time.
    
    active = list(range(R))
    
    def _check_active(active_set):
        """Check if the active terms can reproduce the target."""
        if len(active_set) == 0:
            return False, None
        # Solve constraint_matrix[:, active_set] @ x = target_vec
        A = constraint_matrix[:, active_set]
        try:
            # Use least squares as a check
            x, residuals, rank_, sv = np.linalg.lstsq(A, target_vec, rcond=None)
            if len(residuals) > 0 and residuals[0] > 1e-8:
                return False, None
            # Verify
            recon = A @ x
            if not np.allclose(recon, target_vec, atol=1e-8):
                return False, None
            return True, x
        except:
            return False, None
    
    # Try removing terms one at a time
    for r in range(R):
        if time.time() - start > time_limit:
            break
        
        if r not in active:
            continue
        
        new_active = [a for a in active if a != r]
        can_reproduce, x = _check_active(new_active)
        
        if can_reproduce:
            active = new_active
            # Update U, V, W to reflect the new coefficients
            # The new decomposition is sum_{r in active} x_r * (u_r * v_r * w_r)
            # So the new U[:, r] = x_r * U[:, r], etc.
            for idx, r_idx in enumerate(new_active):
                U[:, r_idx] = x[idx] * U[:, r_idx]
                V[:, r_idx] = x[idx] * V[:, r_idx]
                W[:, r_idx] = x[idx] * W[:, r_idx]
    
    new_rank = len(active)
    
    # Verify the final result
    T_final = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for r in active:
        T_final += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    if np.allclose(T_final, T, atol=1e-8):
        # Return only the active columns
        U_new = U[:, active]
        V_new = V[:, active]
        W_new = W[:, active]
        return U_new, V_new, W_new, new_rank
    
    # If reduction failed, return original
    return U, V, W, R

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses Strassen on 2x2x2 sub-blocks, then attempts rank reduction.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12

    # Build the target tensor for verification
    T = _build_tensor(n, m, p)

    # Try Strassen rank-42 decomposition
    U, V, W, rank = _construct_strassen_2x2x4(n, m, p)
    
    # Verify the Strassen decomposition
    if not _verify(U, V, W, T):
        # Fallback to schoolbook
        U, V, W, rank = _construct_schoolbook(n, m, p)
    else:
        # Attempt rank reduction
        remaining_time = deadline - time.time()
        if remaining_time > 5:  # Only if we have time
            U, V, W, rank = _try_rank_reduction(U, V, W, T, time_limit=max(1, remaining_time - 10))
        
        # Final verification
        if not _verify(U, V, W, T):
            # Fallback to schoolbook
            U, V, W, rank = _construct_schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
