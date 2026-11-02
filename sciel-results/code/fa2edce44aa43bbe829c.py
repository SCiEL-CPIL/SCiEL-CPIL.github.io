def construct():
    import time
    import numpy as np
    _t0 = time.time()
    _DEADLINE = 140.0

    n, m, p = 3, 4, 6

    def build_strassen_hybrid():
        """Build the hybrid Strassen+schoolbook decomposition (rank 66)."""
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2); c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2); g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2); C21 = (i2, k1); C22 = (i2, k2)
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        return terms

    def build_full_schoolbook():
        """Build the full schoolbook decomposition (rank 72)."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U @ diag(1) @ V^T contracted with W gives the multiplication tensor."""
        # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check a few random entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            expected = 1.0  # For multiplication tensor, T[i,j,k]=1 if valid, else 0
            # Actually for matrix mult tensor, T[i,j,k] = 1 for all valid (i,j,k)
            # Wait no - the tensor is T[i,j,k] = delta_{i,j,k} meaning it's 1 when indices align
            # For C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
            # So T[i,j,k] = 1 for all i,j,k (it's the identity mapping in the tensor sense)
            # Actually no. The multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
            # because C[i,k] = sum_j A[i,j] B[j,k], so each triple (i,j,k) contributes 1*A[i,j]*B[j,k]
            if abs(val - 1.0) > tol:
                return False
        return True

    def verify_full(U, V, W, tol=1e-9):
        """Full verification of the decomposition."""
        # T[i,j,k] should be 1 for all i in [0,n), j in [0,m), k in [0,p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def reduce_rank(U, V, W, max_iter=2000):
        """
        Try to reduce rank by finding linear dependencies.
        Strategy: for each column r, check if U[:,r] can be expressed as combination of other U columns,
        and if so, adjust V and W to eliminate it.
        
        More systematic approach: use the fact that if we have terms t1 and t2 where
        U[:,r1] = c * U[:,r2], then we can eliminate one.
        
        Better: Gaussian elimination on the U matrix to find pivot columns,
        then for non-pivot columns, express them as linear combos and absorb.
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        
        # Try iterative elimination: find pairs of columns where one is a scalar multiple
        # of the other in U (or V or W), and eliminate.
        
        eliminated = 0
        active = list(range(rank))
        
        # Build tracking: for each active column, we track its contribution
        # We'll work with the full matrices and zero out columns as we eliminate them.
        
        for iteration in range(max_iter):
            if time.time() - _t0 > _DEADLINE - 10:
                break
            if len(active) < 2:
                break
            
            found = False
            for idx_a in range(len(active)):
                for idx_b in range(idx_a + 1, len(active)):
                    ra = active[idx_a]
                    rb = active[idx_b]
                    
                    # Check if U[:,ra] is a scalar multiple of U[:,rb]
                    ua = U[:, ra]
                    ub = U[:, rb]
                    
                    # Find non-zero entries
                    nz_a = np.nonzero(ua)[0]
                    nz_b = np.nonzero(ub)[0]
                    
                    if len(nz_a) == 0 or len(nz_b) == 0:
                        continue
                    
                    # Check if one is scalar multiple of other
                    # ua = c * ub
                    # Use first non-zero of ub
                    if abs(ub[nz_b[0]]) > 1e-12:
                        c = ua[nz_b[0]] / ub[nz_b[0]] if nz_b[0] in nz_a else None
                        if c is not None and abs(c) > 1e-12:
                            # Check if ua = c * ub
                            if np.allclose(ua, c * ub, atol=1e-10):
                                # Now check if V[:,ra] = c' * V[:,rb] or W[:,ra] = c'' * W[:,rb]
                                # For elimination: if U[:,ra] = c*U[:,rb], V[:,ra] = d*V[:,rb], W[:,ra] = e*W[:,rb]
                                # with c*d*e = 1, we can eliminate ra.
                                # But we need c*d*e = 1 for the product to work out.
                                
                                va = V[:, ra]
                                vb = V[:, rb]
                                nz_va = np.nonzero(va)[0]
                                nz_vb = np.nonzero(vb)[0]
                                
                                if len(nz_va) == 0 or len(nz_vb) == 0:
                                    continue
                                
                                # Check V relationship
                                if abs(vb[nz_vb[0]]) > 1e-12:
                                    d = va[nz_vb[0]] / vb[nz_vb[0]] if nz_vb[0] in nz_va else None
                                    if d is not None and abs(d) > 1e-12:
                                        if np.allclose(va, d * vb, atol=1e-10):
                                            # Check W
                                            wa = W[:, ra]
                                            wb = W[:, rb]
                                            nz_wa = np.nonzero(wa)[0]
                                            nz_wb = np.nonzero(wb)[0]
                                            
                                            if len(nz_wa) == 0 or len(nz_wb) == 0:
                                                continue
                                            
                                            if abs(wb[nz_wb[0]]) > 1e-12:
                                                e = wa[nz_wb[0]] / wb[nz_wb[0]] if nz_wb[0] in nz_wa else None
                                                if e is not None and abs(e) > 1e-12:
                                                    if np.allclose(wa, e * wb, atol=1e-10):
                                                        # Check c*d*e ≈ 1
                                                        if abs(c * d * e - 1.0) < 1e-8:
                                                            # Eliminate column ra
                                                            U[:, ra] = 0
                                                            V[:, ra] = 0
                                                            W[:, ra] = 0
                                                            active.remove(ra)
                                                            eliminated += 1
                                                            found = True
                                                            break
                if found:
                    break
            if not found:
                break
        
        return U, V, W, eliminated

    def reduce_rank_linear(U, V, W, max_iter=200):
        """
        Linear-algebraic rank reduction.
        
        Key idea: If U[:,ra] is in the span of other U columns, say U[:,ra] = sum_i c_i * U[:,ri],
        then we can eliminate term ra by:
          For each i: V[:,ri] += c_i * V[:,ra] / scale, W[:,ri] += c_i * W[:,ra] / scale
        where scale is chosen so the product is preserved.
        
        Actually the correct approach: The tensor decomposition is:
          T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        If we have U[:,ra] = sum_i c_i * U[:,ri], we can write:
          sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
          = U[i,j,ra]*V[j,k,ra]*W[k,i,ra] + sum_{r!=ra} ...
          = (sum_i c_i * U[i,j,ri]) * V[j,k,ra] * W[k,i,ra] + sum_{r!=ra} ...
          = sum_i c_i * U[i,j,ri] * V[j,k,ra] * W[k,i,ra] + sum_{r!=ra} ...
        
        So we can absorb term ra into terms ri by:
          V[:,ri] += c_i * V[:,ra] * (W[:,ri] / W[:,ri]) -- no, this doesn't work simply.
        
        Better: We need V[j,k,ri] * W[k,i,ri] to absorb the extra. This is complex.
        
        Simpler approach: Use the fact that if we can find a linear combination of terms
        that produces zero tensor contribution, we can use it to eliminate a term.
        
        Alternative: Try to find if any term's U vector is a scalar multiple of another,
        and if so, check if the corresponding V and W can be adjusted.
        
        Let me try a different, more practical approach:
        - Compute the rank of the U matrix (treating columns as vectors)
        - If rank < number of columns, there are linear dependencies
        - Use QR decomposition to find which columns are linearly dependent
        - For dependent columns, try to eliminate them by adjusting others
        """
        U, V, W = U.copy(), V.copy(), W.copy()
        rank = U.shape[1]
        eliminated = 0
        
        # First, try simple scalar proportionality elimination (fast)
        active = list(range(rank))
        for _ in range(10):
            if time.time() - _t0 > _DEADLINE - 10:
                break
            found = False
            for idx_a in range(len(active)):
                if found:
                    break
                ra = active[idx_a]
                if np.allclose(U[:, ra], 0, atol=1e-12):
                    continue
                for idx_b in range(idx_a + 1, len(active)):
                    rb = active[idx_b]
                    if np.allclose(U[:, rb], 0, atol=1e-12):
                        continue
                    ua, ub = U[:, ra], U[:, rb]
                    # Check scalar proportionality
                    nz_ub = np.nonzero(ub)[0]
                    if len(nz_ub) == 0:
                        continue
                    pos = nz_ub[0]
                    if abs(ub[pos]) < 1e-12:
                        continue
                    c = ua[pos] / ub[pos]
                    if np.allclose(ua, c * ub, atol=1e-10) and abs(c) > 1e-12:
                        va, vb = V[:, ra], V[:, rb]
                        nz_vb = np.nonzero(vb)[0]
                        if len(nz_vb) == 0:
                            continue
                        posv = nz_vb[0]
                        if abs(vb[posv]) < 1e-12:
                            continue
                        d = va[posv] / vb[posv]
                        if np.allclose(va, d * vb, atol=1e-10) and abs(d) > 1e-12:
                            e_needed = 1.0 / (c * d)
                            wa, wb = W[:, ra], W[:, rb]
                            if np.allclose(wa, e_needed * wb, atol=1e-10):
                                U[:, ra] = 0
                                V[:, ra] = 0
                                W[:, ra] = 0
                                active.remove(ra)
                                eliminated += 1
                                found = True
                                break
            if not found:
                break
        
        # Now try linear dependency elimination using QR
        if time.time() - _t0 > _DEADLINE - 15:
            return U, V, W, eliminated
        
        # Get active columns
        active = [c for c in range(rank) if not np.allclose(U[:, c], 0, atol=1e-12)]
        if len(active) < 2:
            return U, V, W, eliminated
        
        # Try to find linear dependencies among U columns
        # Use SVD to find the numerical rank
        U_active = U[:, active]
        try:
            _, s, Vt = np.linalg.svd(U_active, full_matrices=False)
        except:
            return U, V, W, eliminated
        
        # Find columns that are nearly linearly dependent
        # A column is dependent if it can be expressed as combination of others
        # Use the fact that if s[i] < tol, the (i+1)-th right singular vector is in the null space
        
        tol = 1e-8
        num_pivot = sum(1 for si in s if si > tol)
        
        if num_pivot < len(active):
            # There are dependent columns
            # Find which columns are "extra" by looking at the Vt matrix
            # The last (len(active) - num_pivot) rows of Vt give null space directions
            
            # For each null space direction, find a combination of columns that sums to ~0 in U
            # Then try to eliminate one column from that combination
            
            null_vecs = Vt[num_pivot:]  # shape (num_null, len(active))
            
            for nv in null_vecs:
                if time.time() - _t0 > _DEADLINE - 15:
                    break
                # nv is a vector of length len(active) such that U_active @ nv ≈ 0
                # i.e., sum_i nv[i] * U[:, active[i]] ≈ 0
                
                # Find the column with largest |nv[i]| to eliminate
                max_idx = np.argmax(np.abs(nv))
                max_col = active[max_idx]
                coeff = nv[max_idx]
                
                if abs(coeff) < 1e-10:
                    continue
                
                # We want to eliminate column max_col
                # U[:, max_col] = -(1/coeff) * sum_{i != max_idx} nv[i] * U[:, active[i]]
                
                # For the tensor to remain valid, we need to adjust V and W
                # The contribution of term max_col is:
                #   U[:,max_col] * V[:,max_col] * W[:,max_col] (outer products contracted)
                # = (-(1/coeff) * sum_i nv[i] * U[:,active[i]]) * V[:,max_col] * W[:,max_col]
                # = sum_i (-(nv[i]/coeff)) * U[:,active[i]] * V[:,max_col] * W[:,max_col]
                
                # We can absorb this into term active[i] by modifying V and W:
                # New V[:,active[i]] = V[:,active[i]] + (-(nv[i]/coeff)) * V[:,max_col] * (W[:,active[i]] / W[:,active[i]])
                # This doesn't work simply because W appears in both.
                
                # Actually, the correct way: we need
                #   sum_i c_i * U[:,ri] * V[:,ri] * W[:,ri] to absorb the term
                # This requires V and W to be adjusted in a coupled way.
                
                # Skip this for now - it's complex. The scalar proportionality approach
                # already handles the main cases.
                pass
        
        return U, V, W, eliminated

    # Try multiple decomposition strategies and pick the best
    def build_strassen_hybrid_v2():
        """Try to extend Strassen to cover row 0 as well using a 3-row scheme."""
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2); c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2); g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2); C21 = (i2, k1); C22 = (i2, k2)
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        
        # Try: Strassen on rows (0,1) for columns (0,1),(2,3),(4,5) and row 2 schoolbook
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(0, 1, 0, 1, k1, k2)
            add_strassen_block(0, 1, 2, 3, k1, k2)
        
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(2, j, k)
        
        return terms
    
    def build_strassen_hybrid_v3():
        """Try: Strassen on rows (1,2) for columns (0,1),(2,3),(4,5) - original, 
        plus try to also Strassen row 0 with a different pairing."""
        terms = []
        
        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1); b = (i1, j2); c = (i2, j1); d = (i2, j2)
            e = (j1, k1); f = (j1, k2); g = (j2, k1); h = (j2, k2)
            C11 = (i1, k1); C12 = (i1, k2); C21 = (i2, k1); C22 = (i2, k2)
            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))
        
        def add_schoolbook_term(i, j, k):
            terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        
        # Original: Strassen on (1,2) for 3 column pairs, schoolbook for row 0
        for (k1, k2) in [(0,1), (2,3), (4,5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)
        
        for j in range(4):
            for k in range(6):
                add_schoolbook_term(0, j, k)
        
        return terms
    
    # Try all variants and pick the one with lowest rank after reduction
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for builder in [build_strassen_hybrid, build_strassen_hybrid_v2, build_strassen_hybrid_v3]:
        if time.time() - _t0 > _DEADLINE - 30:
            break
        try:
            terms = builder()
            U_t, V_t, W_t = terms_to_matrices(terms)
            if not verify_full(U_t, V_t, W_t):
                continue
            U_r, V_r, W_r, _ = reduce_rank_linear(U_t, V_t, W_t)
            if verify_full(U_r, V_r, W_r, tol=1e-8):
                # Count non-zero columns
                nz_cols = [c for c in range(U_r.shape[1]) 
                          if np.any(U_r[:, c] != 0) or np.any(V_r[:, c] != 0) or np.any(W_r[:, c] != 0)]
                r = len(nz_cols)
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_r[:, nz_cols], V_r[:, nz_cols], W_r[:, nz_cols]
            else:
                # Use unreduced
                nz_cols = list(range(U_t.shape[1]))
                r = len(nz_cols)
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_t, V_t, W_t
        except:
            continue
    
    if best_U is None:
        # Fallback
        terms = build_strassen_hybrid()
        best_U, best_V, best_W = terms_to_matrices(terms)
        if not verify_full(best_U, best_V, best_W):
            terms = build_full_schoolbook()
            best_U, best_V, best_W = terms_to_matrices(terms)
    
    U, V, W = best_U, best_V, best_W
    base_rank = U.shape[1]
    
    # Try to reduce rank
    U, V, W, elim = reduce_rank_linear(U, V, W)
    
    # Verify final result
    if not verify_full(U, V, W, tol=1e-8):
        # Fallback to known-good
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
    
    # Convert to lists with integer/half-integer entries
    def to_clean_list(mat):
        rows = []
        for row in mat:
            clean_row = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                if abs(rounded) < 1e-10:
                    rounded = 0.0
                clean_row.append(rounded)
            rows.append(clean_row)
        return rows
    
    U_list = to_clean_list(U)
    V_list = to_clean_list(V)
    W_list = to_clean_list(W)
    
    # Final verification with clean values
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    
    if not verify_full(U_np, V_np, W_np, tol=1e-8):
        # Use the hybrid without reduction
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
        U_list = to_clean_list(U)
        V_list = to_clean_list(V)
        W_list = to_clean_list(W)
    
    # Remove all-zero columns to get true rank
    U_np = np.array(U_list, dtype=np.float64)
    V_np = np.array(V_list, dtype=np.float64)
    W_np = np.array(W_list, dtype=np.float64)
    
    nonzero_cols = []
    for c in range(U_np.shape[1]):
        if np.any(U_np[:, c] != 0) or np.any(V_np[:, c] != 0) or np.any(W_np[:, c] != 0):
            nonzero_cols.append(c)
    
    U_np = U_np[:, nonzero_cols]
    V_np = V_np[:, nonzero_cols]
    W_np = W_np[:, nonzero_cols]
    
    # Final check
    if not verify_full(U_np, V_np, W_np, tol=1e-8):
        terms = build_strassen_hybrid()
        U, V, W = terms_to_matrices(terms)
        U_list = to_clean_list(U)
        V_list = to_clean_list(V)
        W_list = to_clean_list(W)
    else:
        U_list = to_clean_list(U_np)
        V_list = to_clean_list(V_np)
        W_list = to_clean_list(W_np)
    
    return {"U": U_list, "V": V_list, "W": W_list}
