<<<<<<< SEARCH
    # Randomized linear algebraic reduction:
    # Try to find linearly dependent rank-1 terms by checking if the
    # tensor can be expressed with fewer rank-1 components.
    # We use multiple random projections to detect dependencies.
    np.random.seed(42)
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For a random vector x (size m*p) and y (size n*m), the contraction
    # T contracted with x and y gives a matrix of size p*n x ... 
    # Actually, let's use the flattening approach:
    # For each rank-1 term r, we have a tensor M_r of size (n*m) x (m*p) x (p*n)
    # where M_r[i,j,k] = U[i,r]*V[j,r]*W[k,r]
    # 
    # A better approach: check if columns of the "coefficient matrix" are dependent.
    # The tensor entries are T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # 
    # Let's form the "flattened" system. For fixed random vectors a (n*m) and b (m*p):
    # S[k] = sum_{i,j} a[i]*b[j]*T[i,j,k] = sum_r (sum_i a[i]*U[i,r]) * (sum_j b[j]*V[j,r]) * W[k,r]
    # 
    # This gives us a vector S of length p*n. If we do this for multiple (a,b) pairs,
    # we get a system that can reveal dependencies.
    #
    # Alternative: directly check if the 66 rank-1 tensors are linearly independent.
    # Flatten each rank-1 tensor to a vector of length n*m*p = 72.
    # If the 66x72 matrix has rank < 66, there are dependencies.
    
    # Build the 66x72 matrix where each row is a flattened rank-1 tensor
    # M[r, i*m*p + j*p + k] = U[i,r] * V[j,r] * W[k,r]
    # This is expensive but doable: 66 * 72 = 4752 entries
    
    # Actually, let's use a smarter approach. The tensor T has entries:
    # T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    #
    # For dependency detection, we can check: are the columns of the "slice" matrices dependent?
    # 
    # Let's try: form matrix A where A[i,k] = sum_j V[j,r]*T[i,j,k] for a random V
    # No, let me think differently.
    #
    # The key insight: if there's a linear dependence sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0,
    # then for ALL i,j,k: sum_r c_r * U[i,r]*V[j,r]*W[k,r] = 0
    # 
    # This means: for each k, sum_r c_r * W[k,r] * (u_r ⊗ v_r)[i,j] = 0
    # i.e., sum_r c_r * W[k,r] * (U[:,r] ⊗ V[:,r]) = 0 for all k
    #
    # So if we form the (n*m*p*n) x rank matrix where entry (i,j,k, r) = U[i,r]*V[j,r]*W[k,r],
    # and check if its column space has dimension < rank, we find dependencies.
    # But that's 72*66 = 4752 entries which is fine.
    
    # Build matrix: rows indexed by (i,j,k) flattened, columns by r
    # M[i*m*p + j*p + k, r] = U[i,r] * V[j,r] * W[k,r]
    M = np.zeros((n * m * p * n, rank), dtype=float)
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                for r in range(rank):
                    M[i * m * p * n + j * p * n + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Compute rank of M
    # Use SVD for numerical stability
    try:
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        
        # Use randomized SVD for speed, or just full SVD since matrix is small (72 x 66)
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        num_independent = np.sum(S > tol)
        
        if num_independent < rank:
            # There are linear dependencies! Find the null space
            # The null space of M^T gives coefficients c such that M^T c = 0
            # i.e., sum_r c[r] * (rank-1 term r) = 0
            # 
            # Actually, we need the left null space: vectors c such that c^T M = 0
            # i.e., M^T c = 0
            # The null space of M^T is spanned by the last (rank - num_independent) columns of Vt^T
            # Wait: M is (72, 66). M^T is (66, 72). 
            # Null space of M^T: vectors c in R^66 such that M^T c = 0.
            # From SVD: M = U_svd @ diag(S) @ Vt
            # M^T = Vt^T @ diag(S) @ U_svd^T
            # Null space of M^T is spanned by rows of Vt corresponding to zero singular values
            # i.e., the last (66 - num_independent) rows of Vt
            
            null_dim = rank - num_independent
            if null_dim > 0:
                # Get null space basis
                null_basis = Vt[num_independent:, :].T  # shape (66, null_dim)
                
                # We can eliminate one column for each null space vector
                # Strategy: for each null vector c, find the column with largest |c[r]|
                # and eliminate it (set its coefficient to 0 by absorbing into others)
                
                # Actually, the correct approach: if sum_r c_r * (term_r) = 0,
                # we can remove term_j where j = argmax |c_r| and adjust others.
                # But we need to be careful: we want to remove a column entirely.
                
                # Better approach: use Gaussian elimination on the column space.
                # Find a maximal linearly independent subset of the 66 columns.
                
                # Use QR with column pivoting
                Q, R, piv = np.linalg.qr(M, pivoting=True)
                # The first num_independent columns (after pivoting) are independent
                # The rest are dependent
                independent_cols = list(piv[:num_independent])
                dependent_cols = list(piv[num_independent:])
                
                # Remove dependent columns
                keep_cols = set(independent_cols)
                new_rank = len(independent_cols)
                
                U_new = [U[i][r] for i in range(n*m) for r in keep_cols] if False else None
                # Reshape properly
                U_new = np.zeros((n*m, new_rank), dtype=float)
                V_new = np.zeros((m*p, new_rank), dtype=float)
                W_new = np.zeros((p*n, new_rank), dtype=float)
                
                for new_r, old_r in enumerate(independent_cols):
                    U_new[:, new_r] = U_np[:, old_r]
                    V_new[:, new_r] = V_np[:, old_r]
                    W_new[:, new_r] = W_np[:, old_r]
                
                # Verify the decomposition is still exact
                # Reconstruct tensor from new decomposition and compare
                T_orig = np.zeros((n*m, m*p, p*n), dtype=float)
                T_new = np.zeros((n*m, m*p, p*n), dtype=float)
                
                for r in range(rank):
                    T_orig += np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
                for r in range(new_rank):
                    T_new += np.outer(np.outer(U_new[:, r], V_new[:, r]), W_new[:, r])
                
                if np.allclose(T_orig, T_new, atol=1e-9):
                    # Success! We reduced the rank
                    # Convert back to list format with integer/half-integer entries
                    U_list = [[int(round(U_new[i, r])) if abs(U_new[i, r] - round(U_new[i, r])) < 1e-9 else 
                              (int(2*U_new[i, r])/2 if abs(2*U_new[i, r] - round(2*U_new[i, r])) < 1e-9 else U_new[i, r])
                              for r in range(new_rank)] for i in range(n*m)]
                    V_list = [[int(round(V_new[i, r])) if abs(V_new[i, r] - round(V_new[i, r])) < 1e-9 else 
                              (int(2*V_new[i, r])/2 if abs(2*V_new[i, r] - round(2*V_new[i, r])) < 1e-9 else V_new[i, r])
                              for r in range(new_rank)] for i in range(m*p)]
                    W_list = [[int(round(W_new[i, r])) if abs(W_new[i, r] - round(W_new[i, r])) < 1e-9 else 
                              (int(2*W_new[i, r])/2 if abs(2*W_new[i, r] - round(2*W_new[i, r])) < 1e-9 else W_new[i, r])
                              for r in range(new_rank)] for i in range(p*n)]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
                
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
=======
    # Randomized linear algebraic reduction to find linearly dependent rank-1 terms
    try:
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build matrix M where M[(i,j,k), r] = U[i,r]*V[j,r]*W[k,r]
        # Shape: (n*m * m*p * p*n, rank) = (72, 66)
        total_entries = n * m * m * p * p * n  # 3*4 * 4*6 * 6*3 = 12*24*18 = 5184
        # Actually: i in [0,12), j in [0,24), k in [0,18)
        # Total = 12*24*18 = 5184
        M = np.zeros((n * m * m * p * p * n, rank), dtype=float)
        
        idx = 0
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    M[idx, :] = U_np[i, :] * V_np[j, :] * W_np[k, :]
                    idx += 1
        
        # SVD to find rank
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        num_independent = int(np.sum(S > tol))
        
        if num_independent < rank:
            # Find independent columns via QR with pivoting
            Q, R, piv = np.linalg.qr(M, pivoting=True)
            independent_cols = list(piv[:num_independent])
            new_rank = num_independent
            
            U_new = U_np[:, independent_cols]
            V_new = V_np[:, independent_cols]
            W_new = W_np[:, independent_cols]
            
            # Verify exactness
            T_orig = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
            T_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            
            if np.allclose(T_orig, T_new, atol=1e-9):
                # Convert to list format
                def to_list(arr):
                    rows, cols = arr.shape
                    result = []
                    for i in range(rows):
                        row = []
                        for r in range(cols):
                            val = arr[i, r]
                            # Check if integer
                            if abs(val - round(val)) < 1e-9:
                                row.append(int(round(val)))
                            # Check if half-integer
                            elif abs(2*val - round(2*val)) < 1e-9:
                                row.append(int(round(2*val)) / 2.0)
                            else:
                                row.append(val)
                        result.append(row)
                    return result
                
                return {"U": to_list(U_new), "V": to_list(V_new), "W": to_list(W_new)}
        
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
