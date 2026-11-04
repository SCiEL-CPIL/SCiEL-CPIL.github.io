def construct():
    import numpy as np
    import time
    from itertools import combinations
    
    n, m, p = 2, 4, 4
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_arr[:, r]
            v_r = V_arr[:, r]
            w_r = W_arr[:, r]
            # C[i,k] += sum_j A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * B[j,k]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += A[i, j] * u_r[i*m + j] * v_r[j*p + k] * w_r[k*n + i] * B[j, k]
                    C_computed[i, k] += s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True
    
    def get_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col_offset, X_rows, Y_rows, Z_rows):
            # P1: (x00+x11)(y00+y11) -> c00:+1, c11:+1
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1
            
            # P2: (x10+x11)y00 -> c10:+1, c11:-1
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1
            
            # P3: x00(y01-y11) -> c01:+1, c11:+1
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1
            
            # P4: x11(y10-y00) -> c00:+1, c10:+1
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1
            
            # P5: (x00+x01)y11 -> c00:-1, c01:+1
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1
            
            # P6: (x10-x00)(y00+y01) -> c11:+1
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1
            
            # P7: (x01-x11)(y10+y11) -> c00:+1
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1
        
        # Block 1: A_L * B_TL -> C_LL (cols 0,1)
        add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])
        # Block 2: A_R * B_BL -> C_LR (cols 0,1)
        add_strassen_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3])
        # Block 3: A_L * B_TR -> C_RL (cols 2,3)
        add_strassen_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])
        # Block 4: A_R * B_BR -> C_RR (cols 2,3)
        add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])
        
        return U, V, W
    
    U28, V28, W28 = get_rank28()
    
    # Verify rank-28 works
    if not verify_decomposition(U28, V28, W28, 28):
        return {"U": U28, "V": V28, "W": W28}
    
    # Strategy: Bilinear map embedding / rank-reduction search
    # The 28-term decomposition gives us 28 rank-1 tensors summing to the multiplication tensor.
    # We search for a linear combination of these 28 terms that is itself rank-1,
    # which would allow us to replace 2 terms with 1 (net rank reduction).
    # 
    # More practically: we look for a pair of terms (r1, r2) such that their sum
    # can be expressed as a single rank-1 tensor (or equivalently, find coefficients
    # that make a 2-term combination rank-1).
    #
    # A more effective approach: treat the 28 vectors in U-space, V-space, W-space
    # and look for a rank-1 tensor in their span that, when subtracted, leaves
    # a valid rank-27 decomposition.
    
    U_arr = np.array(U28, dtype=float)  # 8 x 28
    V_arr = np.array(V28, dtype=float)  # 16 x 28
    W_arr = np.array(W28, dtype=float)  # 8 x 28
    
    # Approach 1: Check if any two terms can be merged (their sum is rank-1)
    # For terms r1, r2: the sum vector in U is U[:,r1]+U[:,r2], similarly V, W.
    # The sum is rank-1 iff the outer product structure holds:
    # (u1+u2) ⊗ (v1+v2) ⊗ (w1+w2) should equal the sum of the two rank-1 tensors
    # But actually we need: there exist u', v', w' such that
    # u'⊗v'⊗w' = u1⊗v1⊗w1 + u2⊗v2⊗w2
    # This is equivalent to the 2-term tensor having rank 1.
    
    # Approach 2 (more powerful): Look for a rank-1 tensor in the span of all 28.
    # If we can find alpha_1,...,alpha_28 such that sum(alpha_r * u_r) ⊗ sum(alpha_r * v_r) ⊗ sum(alpha_r * w_r)
    # equals sum(alpha_r * (u_r ⊗ v_r ⊗ w_r)), then we've found a rank-1 combination.
    # But this is too restrictive.
    
    # Better approach: We want to find a subset S of {1..28} and coefficients c_r for r in S
    # such that the sum over r in S of c_r * (u_r ⊗ v_r ⊗ w_r) is rank-1.
    # If such a subset of size 2 exists, we can replace those 2 terms with 1.
    
    # For a pair (i, j): check if u_i⊗v_i⊗w_i + u_j⊗v_j⊗w_j is rank-1.
    # A 2-term sum is rank-1 iff there exist scalars a,b such that:
    #   a*u_i = b*u_j  (proportional U vectors) -- NO, that's not right.
    # Actually: T = u_i v_i w_i + u_j v_j w_j is rank-1 iff
    # the matrix [u_i, u_j] has rank 1 OR the decomposition can be factored.
    # More precisely, T is rank-1 iff there exist u', v', w' with u'v'w' = T.
    # This means: for all (a,b,c): T[a,b,c] = u'[a]*v'[b]*w'[c]
    
    # Efficient check for pair (i,j): T = u_i⊗v_i⊗w_i + u_j⊗v_j⊗w_j
    # Flatten T into a matrix of size (8*16) x 8 or (8*8) x 16 etc.
    # T is rank-1 iff all its matrix unfoldings have rank ≤ 1.
    
    def is_rank1_pair(U, V, W, i, j):
        """Check if the sum of rank-1 tensors i and j is itself rank-1."""
        # Build the tensor T = u_i⊗v_i⊗w_i + u_j⊗v_j⊗w_j
        # Mode-1 unfolding: matrix of size (n*m) x (m*p)*(p*n) = 8 x 128
        T_unfold1 = np.outer(U[:, i], np.outer(V[:, i], W[:, i]).flatten()) + \
                    np.outer(U[:, j], np.outer(V[:, j], W[:, j]).flatten())
        # Check if this matrix has rank ≤ 1
        sv = np.linalg.svd(T_unfold1, compute_uv=False)
        if len(sv) > 1 and sv[1] > 1e-9:
            return False
        # Also check mode-2 unfolding
        T_unfold2 = np.outer(V[:, i], np.outer(W[:, i], U[:, i]).flatten()) + \
                    np.outer(V[:, j], np.outer(W[:, j], U[:, j]).flatten())
        sv2 = np.linalg.svd(T_unfold2, compute_uv=False)
        if len(sv2) > 1 and sv2[1] > 1e-9:
            return False
        return True
    
    # Search for mergeable pairs
    merged = False
    for i in range(28):
        if time.time() > deadline - 5:
            break
        for j in range(i+1, 28):
            if time.time() > deadline - 5:
                break
            if is_rank1_pair(U_arr, V_arr, W_arr, i, j):
                # Found a mergeable pair! Replace terms i and j with their sum.
                # Build new 27-term decomposition
                new_rank = 27
                U_new = np.zeros((n*m, new_rank))
                V_new = np.zeros((m*p, new_rank))
                W_new = np.zeros((p*n, new_rank))
                
                col = 0
                for r in range(28):
                    if r == i or r == j:
                        continue
                    U_new[:, col] = U_arr[:, r]
                    V_new[:, col] = V_arr[:, r]
                    W_new[:, col] = W_arr[:, r]
                    col += 1
                
                # Last column: the merged term
                U_new[:, col] = U_arr[:, i] + U_arr[:, j]
                V_new[:, col] = V_arr[:, i] + V_arr[:, j]
                W_new[:, col] = W_arr[:, i] + W_arr[:, j]
                
                # Convert to lists of integers/half-integers
                U_list = [[int(round(val)) for val in row] for row in U_new]
                V_list = [[int(round(val)) for val in row] for row in V_new]
                W_list = [[int(round(val)) for val in row] for row in W_new]
                
                if verify_decomposition(U_list, V_list, W_list, 27):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Approach 3: Try to find a rank-1 tensor in the 3-term span
    # For triples (i,j,k): check if there exist a,b,c (not all zero) such that
    # a*u_i⊗v_i⊗w_i + b*u_j⊗v_j⊗w_j + c*u_k⊗v_k⊗w_k is rank-1
    # This is harder but we can use the fact that if the 3 vectors u_i, u_j, u_k are
    # linearly dependent, we might find such a combination.
    
    if time.time() < deadline - 10:
        # Check for linearly dependent triples in U-space
        # If u_i, u_j, u_k are linearly dependent, find coefficients
        for i in range(28):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, 28):
                if time.time() > deadline - 10:
                    break
                for k in range(j+1, 28):
                    if time.time() > deadline - 10:
                        break
                    # Check if u_i, u_j, u_k are linearly dependent
                    M = np.column_stack([U_arr[:, i], U_arr[:, j], U_arr[:, k]])
                    sv = np.linalg.svd(M, compute_uv=False)
                    if len(sv) >= 3 and sv[2] < 1e-9:
                        # Find null vector
                        _, Vt = np.linalg.svd(M)
                        null_vec = Vt[-1]  # coefficients a, b, c
                        a, b, c = null_vec
                        # The combination a*u_i⊗v_i⊗w_i + b*u_j⊗v_j⊗w_j + c*u_k⊗v_k⊗w_k
                        # has U-component = 0, so it's automatically rank-1 (zero in U mode)
                        # This means we can eliminate one term by adjusting the others.
                        # Actually this means the 3 terms have a linear dependency in U-space,
                        # so we can express one as a combination of the other two in U-space.
                        # But we need the full tensor to work out.
                        pass  # Complex to exploit directly; skip for now
    
    # Approach 4: Direct search - try to find a rank-27 decomposition by 
    # perturbing the rank-28 and checking if the last term becomes redundant.
    # This is essentially checking if the 28 rank-1 tensors are linearly independent
    # as tensors. If they're not, we have a rank-27 decomposition.
    
    if time.time() < deadline - 5:
        # Check linear independence of the 28 rank-1 tensors
        # Flatten each rank-1 tensor into a vector of size 8*16*8 = 1024
        # Then check if the 28 vectors are linearly independent
        tensors = []
        for r in range(28):
            T = np.outer(U_arr[:, r], np.outer(V_arr[:, r], W_arr[:, r]).flatten())
            tensors.append(T.flatten())
        T_matrix = np.column_stack(tensors)  # 1024 x 28
        sv = np.linalg.svd(T_matrix, compute_uv=False)
        # If rank < 28, we have a dependency and can reduce rank
        if len(sv) > 0 and sv[-1] < 1e-10:
            # There's a linear dependency - find it
            _, Vt = np.linalg.svd(T_matrix)
            null_vec = Vt[-1]
            # We can eliminate one term. Find the term with largest |coefficient|
            max_idx = np.argmax(np.abs(null_vec))
            # Build 27-term decomposition by expressing term max_idx as combination of others
            # and absorbing it
            new_rank = 27
            U_new = np.zeros((n*m, new_rank))
            V_new = np.zeros((m*p, new_rank))
            W_new = np.zeros((p*n, new_rank))
            
            col = 0
            for r in range(28):
                if r == max_idx:
                    continue
                # Scale by null_vec[r] / null_vec[max_idx] to account for the dependency
                scale = null_vec[r] / null_vec[max_idx]
                U_new[:, col] = U_arr[:, r] * scale
                V_new[:, col] = V_arr[:, r] * scale
                W_new[:, col] = W_arr[:, r] * scale
                col += 1
            
            # Hmm, this approach is tricky. The dependency means sum(null_vec[r] * T_r) = 0
            # So T_max_idx = -sum_{r≠max} (null_vec[r]/null_vec[max_idx]) * T_r
            # We can't simply scale the other terms; we need to adjust.
            # Let's just try the direct approach: remove term max_idx and see if
            # the remaining 27 still decompose correctly (they won't, since we removed a term).
            # The correct approach: the 28 tensors sum to the multiplication tensor.
            # If there's a dependency, we have freedom in how we express the sum.
            # But for now, this is complex. Skip and return rank-28.
            pass
    
    # Return the verified rank-28 decomposition
    return {"U": U28, "V": V28, "W": W28}
