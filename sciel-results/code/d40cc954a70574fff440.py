<<<<<<< SEARCH
    def try_find_rank32():
        """
        Attempt to find a rank-32 decomposition.
        Strategy 1: Structured search - perturb the standard parts of the rank-33
                    solution and try to absorb one term.
        Strategy 2: Random search with constraints from the tensor structure.
        """
        r = 32
        
        # Strategy 1: Try to merge two of the 12 standard terms by modifying
        # them and adjusting nearby Strassen terms.
        # The standard terms in block b are at columns 7*b+7 .. 7*b+10 (0-indexed).
        # Term indices within block: 0,1,2,3 correspond to (i,k) = (0,0),(0,1),(1,0),(1,1)
        
        # Idea: Can we replace two standard terms (i1,k1) and (i2,k2) in the same
        # block with a single term that, combined with a modified Strassen term,
        # gives the same result?
        
        # This is complex. Let's try a more direct approach:
        # Random search with structure-aware initialization.
        
        np.random.seed(42)
        
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with structure: mostly based on rank-33,
            # but with one column removed and others perturbed.
            
            # Start from rank-33 and try to "fold" one column into others.
            U33, V33, W33 = strassen_block_decomposition()
            
            # Pick a column to eliminate (prefer standard columns as they're simpler)
            # Standard columns are at positions: 7,8,9,10, 14,15,16,17, 21,22,23,24
            std_cols = [7,8,9,10, 14,15,16,17, 21,22,23,24]
            
            if attempt < 12:
                # Try eliminating each standard column
                elim_col = std_cols[attempt]
            else:
                # Try random columns
                elim_col = np.random.randint(0, 33)
            
            # Create rank-32 matrices by removing column elim_col
            mask = np.ones(33, dtype=bool)
            mask[elim_col] = False
            U32 = U33[:, mask].copy()
            V32 = V33[:, mask].copy()
            W32 = W33[:, mask].copy()
            
            # The difference tensor is:
            # D = U33[:,elim_col] * V33[:,elim_col] * W33[:,elim_col]
            # We need to absorb this into the remaining 32 terms.
            
            # This means we need to find vectors u', v', w' such that:
            # sum_{c=0}^{31} (U32[:,c] + du_c) * (V32[:,c] + dv_c) * (W32[:,c] + dw_c) = T
            # where du_c, dv_c, dw_c are small perturbations.
            
            # Linearizing: sum_c [U32[:,c]*dv_c*W32[:,c] + U32[:,c]*V32[:,c]*dw_c + du_c*V32[:,c]*W32[:,c]] = D
            # (ignoring second-order terms)
            
            # This is a linear system in du, dv, dw (3*32*... unknowns, 1296 equations).
            # Too many unknowns. Let's restrict perturbations to specific columns.
            
            # Try perturbing only 1-3 columns.
            if attempt % 3 == 0:
                # Perturb 1 column
                pert_cols = [np.random.randint(0, 32)]
            elif attempt % 3 == 1:
                # Perturb 2 columns
                pert_cols = list(np.random.choice(32, 2, replace=False))
            else:
                # Perturb 3 columns
                pert_cols = list(np.random.choice(32, 3, replace=False))
            
            # For simplicity, try adding a new rank-1 term that approximates D
            # by finding the best rank-1 approximation of D.
            
            D = np.einsum('i,j,k->ijk', U33[:,elim_col], V33[:,elim_col], W33[:,elim_col])
            
            # Find best rank-1 approximation of D: min ||D - u*v*w||_F
            # This is non-trivial. Use SVD-based approach:
            # D1 = D.reshape(dim_a, dim_b*dim_c)
            # U_svd, S_svd, Vt_svd = np.linalg.svd(D1, full_matrices=False)
            # u = U_svd[:,0] * S_svd[0]
            # vw = Vt_svd[0].reshape(dim_b, dim_c)
            # Then factor vw = v*w (another SVD)
            
            D1 = D.reshape(dim_a, dim_b * dim_c)
            U_svd, S_svd, Vt_svd = np.linalg.svd(D1, full_matrices=False)
            if S_svd[0] < 1e-10:
                continue
            u_approx = U_svd[:, 0] * S_svd[0]
            vw_approx = Vt_svd[0].reshape(dim_b, dim_c)
            
            # Factor vw_approx = v_approx * w_approx
            U2_svd, S2_svd, V2t_svd = np.linalg.svd(vw_approx, full_matrices=False)
            if S2_svd[0] < 1e-10:
                continue
            v_approx = U2_svd[:, 0] * S2_svd[0]
            w_approx = V2t_svd[0]
            
            # Add this as a new column? No, we need to REPLACE, not add.
            # So we need to modify existing columns to absorb D.
            
            # Try: add u_approx, v_approx, w_approx to one existing column
            # and check if the result is still valid.
            # This won't work directly because we'd be adding, not replacing.
            
            # Alternative: try random search from scratch with small entries.
            if attempt >= 12:
                U = np.random.randint(-1, 2, size=(dim_a, r))
                V = np.random.randint(-1, 2, size=(dim_b, r))
                W = np.random.randint(-1, 2, size=(dim_c, r))
                
                if verify_decomposition(U, V, W):
                    return U, V, W
        
        return None
=======
    def try_find_rank32():
        """
        Attempt to find a rank-32 decomposition.
        Strategy 1: Check if any rank-1 term in the rank-33 solution is in the
                    span of the others (linear algebra check).
        Strategy 2: Try to find a 2x3x3 rank-16 sub-decomposition.
        Strategy 3: Structured perturbation search.
        """
        r = 32
        U33, V33, W33 = strassen_block_decomposition()
        std_cols = [7,8,9,10, 14,15,16,17, 21,22,23,24]
        
        # Strategy 1: For each column, check if its rank-1 tensor is in the
        # span of the other 32 rank-1 tensors. This is a linear algebra problem.
        # The tensor T = sum_c u_c ⊗ v_c ⊗ w_c. If we remove column e, we need
        # u_e ⊗ v_e ⊗ w_e to be expressible as a combination of the others.
        # This is equivalent to: the 3D array D_e = u_e ⊗ v_e ⊗ w_e must lie
        # in the linear span of {u_c ⊗ v_c ⊗ w_c : c ≠ e}.
        
        # We can check this by building the matrix of all 33 rank-1 tensors
        # (flattened) and checking if removing any one doesn't reduce the rank.
        
        # Flatten all rank-1 tensors into a 1296-dim vector
        rank1_vecs = []
        for c in range(33):
            vec = np.einsum('i,j,k->ijk', U33[:,c], V33[:,c], W33[:,c]).flatten()
            rank1_vecs.append(vec)
        
        # Build 1296 x 33 matrix
        M = np.column_stack(rank1_vecs).astype(np.float64)
        
        # Check: if rank(M) < 33, then there's a linear dependency, meaning
        # one term is in the span of others.
        rank_M = np.linalg.matrix_rank(M, tol=1e-10)
        
        if rank_M < 33:
            # Find which column is dependent
            # Use QR with column pivoting
            Q, R, piv = np.linalg.qr(M, mode='complete', pivoting=True)
            # The last column (or columns where |R[i,i]| is tiny) is dependent
            diag_R = np.abs(np.diag(R[:33, :33]))
            # Find columns with smallest diagonal entries
            order = np.argsort(diag_R)
            dep_col = piv[order[0]]
            
            # Now find the coefficients: express column dep_col as combo of others
            # Solve M[:, [c for c in range(33) if c != dep_col]] * x = M[:, dep_col]
            mask = np.ones(33, dtype=bool)
            mask[dep_col] = False
            A_mat = M[:, mask]
            b_vec = M[:, dep_col]
            
            # Least squares solve
            x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
            
            # Check if this gives an exact solution
            recon = A_mat @ x
            if np.allclose(recon, b_vec, atol=1e-6):
                # We have a linear dependency! But this doesn't directly give us
                # a rank-32 tensor decomposition - it means the TENSOR is the same,
                # but we still need 33 terms unless we can restructure.
                # Actually, if the rank-1 tensors are linearly dependent, the tensor
                # rank might still be 33. We need a different approach.
                pass
        
        # Strategy 2: Try to construct a 2x3x3 rank-16 decomposition.
        # If we can do 2x3x3 in rank 16, then 2x3x6 = 2 * (2x3x3) in rank 32.
        # 
        # 2x3x3: A(2x3), B(3x3), C(2x3)
        # Split B into [B_left(3x2), B_right(3x1)]
        # C = A @ B = [A @ B_left (2x2), A @ B_right (2x1)]
        #
        # For the 2x2 part: A(2x3) @ B_left(3x2) = C_left(2x2)
        #   Split A into [A_top(2x2), A_bot(2x1)]
        #   C_left = A_top @ B_left[:2,:2] + A_bot @ B_left[2,:2]
        #   First term: 2x2 @ 2x2 = 2x2, use Strassen (rank 7)
        #   Second term: 2x1 @ 1x2 = 2x2, rank 4 (schoolbook)
        #   Total: 11
        #
        # For the 2x1 part: A(2x3) @ B_right(3x1) = C_right(2x1)
        #   C_right[i] = sum_j A[i,j] * B_right[j]
        #   This is 2 dot products of length 3, rank 2.
        #
        # Total for 2x3x3: 11 + 2 = 13? That seems too low. Let me verify.
        # Actually the issue is that the Strassen decomposition for 2x2x2 
        # requires specific structure. Let me just try to search computationally.
        
        # Strategy 3: Direct search for rank-32 using the structure.
        # Try all pairs of columns to eliminate and check if the remaining 31
        # can be adjusted (with small integer perturbations) to still decompose T.
        
        np.random.seed(12345)
        
        # First, try a systematic approach: for each standard column, try to
        # absorb it by modifying at most 2 other columns.
        for elim in std_cols:
            if time.time() > deadline - 10:
                break
            
            # The rank-1 tensor to absorb
            D = np.einsum('i,j,k->ijk', U33[:,elim], V33[:,elim], W33[:,elim]).astype(np.float64)
            
            # Try modifying 2 other columns
            other_cols = [c for c in range(33) if c != elim]
            
            # For speed, only try a subset
            for trial in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Pick 2 columns to modify
                if trial < 2:
                    mod_cols = other_cols[trial] if isinstance(other_cols[trial], list) else [other_cols[trial], other_cols[trial+1]]
                else:
                    mod_cols = list(np.random.choice(other_cols, 2, replace=False))
                
                # Try small integer perturbations to these 2 columns
                # and check if we can absorb D.
                # This is: find du1,dv1,dw1,du2,dv2,dw2 such that
                # sum of perturbed terms = D (approximately)
                # Linearize: for small perturbations,
                # U[:,c1]*dv1*W[:,c1] + U[:,c1]*V[:,c1]*dw1 + du1*V[:,c1]*W[:,c1]
                # + U[:,c2]*dv2*W[:,c2] + U[:,c2]*V[:,c2]*dw2 + du2*V[:,c2]*W[:,c2] = D
                
                # Build the linear system
                # Unknowns: du1(6), dv1(18), dw1(12), du2(6), dv2(18), dw2(12) = 72 unknowns
                # Equations: 6*18*12 = 1296
                # Overdetermined, but let's try least squares and check.
                
                A_system = np.zeros((1296, 72))
                for idx, c in enumerate(mod_cols):
                    Uc = U33[:, c].astype(np.float64)
                    Vc = V33[:, c].astype(np.float64)
                    Wc = W33[:, c].astype(np.float64)
                    base = idx * 36
                    
                    # du contribution: du * Vc * Wc
                    for i in range(dim_a):
                        for j in range(dim_b):
                            for k in range(dim_c):
                                A_system[(i*dim_b+j)*dim_c+k, base+i] = Vc[j] * Wc[k]
                    
                    # dv contribution: Uc * dv * Wc
                    for i in range(dim_a):
                        for j in range(dim_b):
                            for k in range(dim_c):
                                A_system[(i*dim_b+j)*dim_c+k, base+dim_a+j] = Uc[i] * Wc[k]
                    
                    # dw contribution: Uc * Vc * dw
                    for i in range(dim_a):
                        for j in range(dim_b):
                            for k in range(dim_c):
                                A_system[(i*dim_b+j)*dim_c+k, base+2*dim_a+k] = Uc[i] * Vc[j]
                
                b_vec = D.flatten()
                
                # Solve least squares
                sol, res, rank_s, sv = np.linalg.lstsq(A_system, b_vec, rcond=1e-10)
                
                # Check residual
                if res[0] < 1e-4:
                    # Check if solution is close to integer
                    sol_int = np.round(sol)
                    recon = A_system @ sol_int
                    if np.allclose(recon, b_vec, atol=1e-6):
                        # Apply perturbations
                        U_new = U33[:, [c for c in range(33) if c != elim]].copy()
                        V_new = V33[:, [c for c in range(33) if c != elim]].copy()
                        W_new = W33[:, [c for c in range(33) if c != elim]].copy()
                        
                        for idx, c in enumerate(mod_cols):
                            new_idx = list(range(33))[list(range(33)).index(c)]
                            # Find index in new arrays
                            new_idx = [i for i, orig in enumerate([c2 for c2 in range(33) if c2 != elim]) if orig == c][0]
                            base = idx * 36
                            U_new[:, new_idx] += sol_int[base:base+dim_a].astype(int)
                            V_new[:, new_idx] += sol_int[base+dim_a:base+2*dim_a].astype(int)
                            W_new[:, new_idx] += sol_int[base+2*dim_a:base+3*dim_a].astype(int)
                        
                        if verify_decomposition(U_new, V_new, W_new):
                            return U_new, V_new, W_new
        
        # Strategy 4: Broader random search with structure
        for attempt in range(2000):
            if time.time() > deadline - 5:
                break
            
            # Start from rank-33, remove a random column, try random integer
            # perturbations to 1-3 other columns
            elim = np.random.randint(0, 33)
            D = np.einsum('i,j,k->ijk', U33[:,elim], V33[:,elim], W33[:,elim]).astype(np.float64)
            
            n_pert = np.random.randint(1, 4)
            other_cols = [c for c in range(33) if c != elim]
            mod_cols = list(np.random.choice(other_cols, n_pert, replace=False))
            
            # Build and solve linear system
            n_unknowns = n_pert * (dim_a + dim_b + dim_c)
            A_system = np.zeros((1296, n_unknowns))
            
            for idx, c in enumerate(mod_cols):
                Uc = U33[:, c].astype(np.float64)
                Vc = V33[:, c].astype(np.float64)
                Wc = W33[:, c].astype(np.float64)
                base = idx * (dim_a + dim_b + dim_c)
                
                # Vectorized: du contribution
                for i in range(dim_a):
                    A_system[:, base+i] = (Vc * Wc).flatten() * Uc[i]
                
                # dv contribution  
                for j in range(dim_b):
                    A_system[:, base+dim_a+j] = (Uc * Wc).flatten() * Vc[j]
                
                # dw contribution
                for k in range(dim_c):
                    A_system[:, base+dim_a+dim_b+k] = (Uc * Vc).flatten() * Wc[k]
            
            b_vec = D.flatten()
            sol, res, rank_s, sv = np.linalg.lstsq(A_system, b_vec, rcond=1e-10)
            
            if res[0] < 1e-3:
                sol_int = np.round(sol)
                recon = A_system @ sol_int
                if np.allclose(recon, b_vec, atol=1e-6):
                    U_new = U33[:, [c for c in range(33) if c != elim]].copy()
                    V_new = V33[:, [c for c in range(33) if c != elim]].copy()
                    W_new = W33[:, [c for c in range(33) if c != elim]].copy()
                    
                    for idx, c in enumerate(mod_cols):
                        new_idx = [i for i, orig in enumerate([c2 for c2 in range(33) if c2 != elim]) if orig == c][0]
                        base = idx * (dim_a + dim_b + dim_c)
                        U_new[:, new_idx] += sol_int[base:base+dim_a].astype(int)
                        V_new[:, new_idx] += sol_int[base+dim_a:base+2*dim_a].astype(int)
                        W_new[:, new_idx] += sol_int[base+2*dim_a:base+3*dim_a].astype(int)
                    
                    if verify_decomposition(U_new, V_new, W_new):
                        return U_new, V_new, W_new
        
        return None
>>>>>>> REPLACE

Now let me also improve the main function to use the time more effectively:

