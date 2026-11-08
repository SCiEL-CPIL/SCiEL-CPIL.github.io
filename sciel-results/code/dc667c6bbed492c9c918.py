<<<<<<< SEARCH
    # Build final matrices
    rank = len(U_cols)
    U = np.array([U_cols[col] for col in range(rank)]).T
    V = np.array([V_cols[col] for col in range(rank)]).T
    W = np.array([W_cols[col] for col in range(rank)]).T
    
    # Attempt rank reduction: find linearly dependent columns and eliminate them
    # The tensor decomposition requires: sum_r U[:,r] * V[:,r] * W[:,r] = T
    # If we find a linear dependency among the "interaction" vectors, we can reduce rank.
    # 
    # A key insight: if two terms r1, r2 have U[:,r1] = a*U[:,r2] and V[:,r1] = b*V[:,r2],
    # then W[:,r1] must be (a*b)*W[:,r2] for the decomposition to be valid.
    # More generally, we can look for rank-reducing transformations.
    
    # Try to find a lower rank by checking if the set of rank-1 tensors is linearly dependent
    # in the tensor space. We form the matrix where each term contributes to the 3-way tensor.
    # For a 4x4*4x9 multiplication, the tensor has size 16*36*36 = 20736 elements.
    # Each rank-1 term is a vector in R^20736. We check if these rank vectors are linearly dependent.
    
    if time.time() < DEADLINE:
        # Build the tensor matrix: each column is one rank-1 term flattened
        # Term r contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We can check linear dependence by forming a matrix with these as columns
        # But 20736 x rank might be too large. Instead, use a random projection to check.
        
        # Alternative: use the fact that if we can find coefficients c_r such that
        # sum_r c_r * U[:,r] ⊗ V[:,r] = 0 AND sum_r c_r * W[:,r] = 0,
        # then we can eliminate one term.
        
        # Simpler approach: check if any two terms are scalar multiples of each other
        # in the (U, V, W) triple sense.
        
        # Try pairwise elimination: for each pair (i, j), check if term j can be
        # expressed as a combination that allows removing it.
        # This is complex. Let's try a different approach:
        
        # Use SVD on the "interaction" structure.
        # Form matrix M where M[i,j] = sum_r U[i,r]*V[j,r] for the first two modes,
        # but we also need W. 
        
        # Practical approach: try to find a term r such that its contribution
        # can be absorbed by modifying other terms.
        
        # For now, try a simple greedy: remove terms that are linearly dependent
        # in the combined (U,V,W) space using a random test vector.
        
        # Generate a random test to check linear independence
        np.random.seed(42)
        
        # Check if the rank-1 tensors are linearly independent by testing
        # with random vectors a (for U), b (for V), c (for W)
        # The bilinear form: <a, U[:,r]> * <b, V[:,r]> * <c, W[:,r]>
        # If we can find non-trivial c_r with sum_r c_r * <a,U[:,r]> * <b,V[:,r]> * <c,W[:,r]> = 0
        # for all a,b,c, then the terms are dependent.
        
        # This is equivalent to checking if the Khatri-Rao product has full column rank.
        # The Khatri-Rao product of U, V, W has size (16*36) x rank for the first two,
        # but we need all three.
        
        # Let's just try to verify the current decomposition is correct and return it.
        # The rank reduction is non-trivial and may not be achievable with simple methods.
        
        # Verify correctness
        def verify(U_mat, V_mat, W_mat):
            n, m, p = DIMENSIONS
            # Compute C = A * B for random A, B
            np.random.seed(123)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                C_computed += U_mat[:, r] * V_mat[:, r] * W_mat[:, r]
                # Actually, the tensor decomposition is:
                # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Let me re-check the convention.
            
            # The correct computation:
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W_mat[k * n + i, r] * (
                            U_mat[:, r] * V_mat[:, r]
                        )[i * m + k]  # This isn't right either
            
            # Let me use the proper tensor contraction:
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # And C[i,k] = sum_j T[i,j,k]
            
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                # For each r, we have vectors u_r (16), v_r (36), w_r (36)
                # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] summed over j gives contribution to C[i,k]
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for j in range(m):
                            s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                        C_computed[i, k] += s
            
            return np.allclose(C_computed, C_expected, atol=1e-10)
        
        if not verify(U, V, W):
            raise ValueError("Decomposition is incorrect!")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # Build final matrices
    rank = len(U_cols)
    U = np.array([U_cols[col] for col in range(rank)]).T
    V = np.array([V_cols[col] for col in range(rank)]).T
    W = np.array([W_cols[col] for col in range(rank)]).T
    
    # Verify correctness
    n, m, p = DIMENSIONS
    def verify_decomp(U_mat, V_mat, W_mat):
        np.random.seed(123)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    C_computed[i, k] += s
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not verify_decomp(U, V, W):
        raise ValueError("Decomposition is incorrect!")
    
    # Attempt rank reduction via local search
    # Try to find a term that can be eliminated by adjusting others
    if time.time() < DEADLINE:
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()
        rank_work = rank
        
        # Greedy elimination: try removing each term and see if the remaining
        # terms can be adjusted to compensate.
        # This is NP-hard in general, but we can try a simple heuristic.
        
        # Alternative: check for exact linear dependencies in the Khatri-Rao sense.
        # Form the matrix K where K[a, r] = U[a, r] * V[b, r] for a fixed b... 
        # This is complex. Let's try a simpler approach.
        
        # Check if any two columns are proportional in all three matrices
        for r1 in range(rank_work):
            if time.time() > DEADLINE:
                break
            for r2 in range(r1 + 1, rank_work):
                # Check if U[:,r1] = alpha * U[:,r2], V[:,r1] = beta * V[:,r2], W[:,r1] = gamma * W[:,r2]
                # and alpha * beta * gamma = 1 (for the tensor to be preserved)
                
                # Find alpha from U
                mask_u = np.abs(U_work[:, r2]) > 1e-10
                if not np.any(mask_u):
                    continue
                alpha_candidates = U_work[mask_u, r1] / U_work[mask_u, r2]
                if not np.allclose(alpha_candidates, alpha_candidates[0], atol=1e-10):
                    continue
                alpha = alpha_candidates[0]
                
                mask_v = np.abs(V_work[:, r2]) > 1e-10
                if not np.any(mask_v):
                    continue
                beta_candidates = V_work[mask_v, r1] / V_work[mask_v, r2]
                if not np.allclose(beta_candidates, beta_candidates[0], atol=1e-10):
                    continue
                beta = beta_candidates[0]
                
                mask_w = np.abs(W_work[:, r2]) > 1e-10
                if not np.any(mask_w):
                    continue
                gamma_candidates = W_work[mask_w, r1] / W_work[mask_w, r2]
                if not np.allclose(gamma_candidates, gamma_candidates[0], atol=1e-10):
                    continue
                gamma = gamma_candidates[0]
                
                # Check if alpha * beta * gamma ≈ 1
                if np.isclose(alpha * beta * gamma, 1.0, atol=1e-10):
                    # Term r1 is redundant, remove it
                    U_work = np.delete(U_work, r1, axis=1)
                    V_work = np.delete(V_work, r1, axis=1)
                    W_work = np.delete(W_work, r1, axis=1)
                    rank_work -= 1
                    break
            else:
                continue
            break
    
    if rank_work < rank:
        if verify_decomp(U_work, V_work, W_work):
            return {"U": U_work.tolist(), "V": V_work.tolist(), "W": W_work.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
