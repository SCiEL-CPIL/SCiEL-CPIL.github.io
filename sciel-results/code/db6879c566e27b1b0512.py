import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n, m, p = 3, 4, 6

    def build_tensor(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        return np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)

    def verify(U, V, W, atol=1e-9):
        R = build_tensor(U, V, W)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
        if not np.allclose(R, target, atol=atol):
            return False
        return True

    def get_strassen_66():
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        # Row 0 of C (1x2 A blocks)
        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        # Rows 1-2 of C (2x2 A blocks, Strassen)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)

        assert col == 66, f"Expected 66 columns, got {col}"
        return U, V, W

    def prune_decomposition(U, V, W, max_iter=200):
        """Iteratively remove linearly dependent terms from the decomposition."""
        current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        current_rank = len(U[0])
        target_tensor = build_tensor(current_U, current_V, current_W)

        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break

            # Build the "matrix" whose rows are the terms (flattened outer products)
            Ua = np.array(current_U, dtype=float)  # (12, rank)
            Va = np.array(current_V, dtype=float)  # (24, rank)
            Wa = np.array(current_W, dtype=float)  # (18, rank)

            # Each term c is a vector in R^{12*24*18}
            # Term c = U[:,c] outer V[:,c] outer W[:,c]
            # We want to find if any term is a linear combination of others.
            
            # Efficient check: for each term j, check if removing it changes the tensor
            # To speed this up, we can first check if the term is "trivially" redundant
            # by looking at the rank of the coefficient vectors.
            
            # Let's use a smarter approach: 
            # The tensor T = sum_c u_c v_c w_c
            # If we fix all but one index, we get a matrix.
            # Specifically, for a fixed (i,k), T[i,k,:] = sum_c U[i,c] V[k,c] W[:,c]
            # This is a sum of rank-1 matrices (scaled W[:,c]).
            
            # A faster redundancy test:
            # Check if term j can be expressed as a linear combo of other terms.
            # This is equivalent to checking if the vector t_j (flattened) is in the span of {t_l : l != j}.
            
            # Instead of full SVD every time (expensive), let's try removing terms one by one
            # starting from the end, and check if the reconstruction still matches.
            # We can optimize the check by only verifying the difference.
            
            removed_any = False
            # Try removing from the last term backwards
            for rem_idx in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                
                # Tentatively remove term rem_idx
                U_new = [row for i, row in enumerate(current_U) if i != rem_idx]
                V_new = [row for i, row in enumerate(current_V) if i != rem_idx]
                W_new = [row for i, row in enumerate(current_W) if i != rem_idx]
                
                if not U_new:
                    continue
                    
                # Quick check: does the new decomposition still equal the target?
                # Compute difference: T_new - T_target
                # T_new = T_target - t_rem
                # So we need t_rem to be 0? No, that's only if it was the only term.
                # We need T_new == T_target.
                # This implies T_target - T_new = 0 => t_rem + (T_target - T_target) ... wait.
                # T_target = sum_all t_c.
                # T_new = sum_{c != rem} t_c.
                # T_target - T_new = t_rem.
                # So T_new == T_target if and only if t_rem == 0?
                # No! t_rem is a fixed vector. The sum of the OTHER terms must equal T_target.
                # This means t_rem must be expressible as a linear combination of the other terms?
                # No, the coefficients are fixed (1). We cannot change coefficients.
                # So we can only remove term j if the sum of the remaining terms equals the target.
                # This happens if and only if the vector t_j is in the span of the other vectors?
                # No, the decomposition is fixed. We are not re-solving coefficients.
                # We are just deleting a term.
                # So we check: is sum_{c != j} t_c == T_target?
                # This is equivalent to: is t_j == 0?
                # Wait, if t_j is not zero, but the sum of others equals T_target, that means
                # T_target = sum_{c != j} t_c.
                # But T_target = sum_{c != j} t_c + t_j.
                # So t_j must be 0.
                
                # UNLESS... we are allowed to adjust the remaining terms?
                # The problem asks for a decomposition. Usually, this means fixed U,V,W matrices.
                # If we remove a column, the remaining columns are fixed.
                # So we can only remove a column if its contribution is zero?
                # Or if the columns are linearly dependent AND we can merge them?
                # But merging them changes the entries of the remaining columns.
                # The prompt says "Find a rank-minimal exact tensor decomposition".
                # It does not say we can modify the existing columns, only that the result must be a valid decomposition.
                # So, if we have terms t1, t2, t3 and t3 = t1 + t2, we can remove t3.
                # The new decomposition is just t1, t2.
                # So we need to check if the vector t_j is in the span of {t_l : l != j}.
                
                # Let's implement this check properly.
                # Construct matrix M where each row is a flattened term.
                # M has shape (rank, 12*24*18).
                # This is 66 * 5184 = 342k floats. Manageable.
                
                if current_rank < 2:
                    break

                # Build the term matrix
                # t_c = U[:,c] outer V[:,c] outer W[:,c]
                # We can compute this efficiently.
                # Actually, we just need to check linear dependence.
                # Let's use the SVD of the matrix of terms.
                
                # Optimization: Only do this if we haven't done it recently or if rank changed.
                # For now, let's just do it every iteration but break early if no removal.
                
                # To avoid rebuilding the huge matrix every time, let's just try removal and verify.
                # Verification is O(rank * 12*24*18) which is similar to building the matrix.
                # But verification uses einsum which is optimized.
                
                Ua_new = np.array(U_new, dtype=float)
                Va_new = np.array(V_new, dtype=float)
                Wa_new = np.array(W_new, dtype=float)
                
                # Compute T_new
                T_new = np.einsum('ic,kc,lc->ikl', Ua_new, Va_new, Wa_new)
                
                # Check against target_tensor
                if np.allclose(T_new, target_tensor, atol=1e-8):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break # Restart loop with new rank
            
            if not removed_any:
                break

        return current_U, current_V, current_W, current_rank

    # Get the base decomposition
    U, V, W = get_strassen_66()
    
    # Verify base
    if not verify(U, V, W):
        # Should not happen, but fallback to schoolbook if broken
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r*m+b
                    k = b*p+c_c
                    l = c_c*n+c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
    
    # Attempt pruning
    U_pruned, V_pruned, W_pruned, r_pruned = prune_decomposition(U, V, W)
    
    # Verify pruned
    if verify(U_pruned, V_pruned, W_pruned):
        return {"U": U_pruned, "V": V_pruned, "W": W_pruned}
    else:
        # Return original if pruning failed or broke validity
        return {"U": U, "V": V, "W": W}

