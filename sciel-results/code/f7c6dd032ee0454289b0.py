<<<<<<< SEARCH
    # ---- Fast column-elimination via tensor-flattening rank test ----
    # M[ijk, l] = U[i m+j, l] * V[j p+k, l] * W[k n+i, l]; the decomposition is
    # valid iff every tensor entry is reproduced, i.e. M @ (col vector) covers ones.
    # A column can be removed iff the remaining columns still span the ones vector.
    def build_M(U_mat, V_mat, W_mat, r):
        M = np.zeros((n * m * p, r))
        for l in range(r):
            colU = U_mat[:, l]
            colV = V_mat[:, l]
            colW = W_mat[:, l]
            for i in range(n):
                base = i * m * p
                for j in range(m):
                    a = colU[i * m + j]
                    if a == 0.0:
                        continue
                    for k in range(p):
                        M[base + j * p + k, l] = a * colV[j * p + k] * colW[k * n + i]
        return M

    def can_reduce(M, target, r, remove_idx):
        # Does the column space of M with column remove_idx dropped still contain target?
        # Equivalent: is target in colspace(M)? removing a column keeps target in
        # colspace only if that column is redundant. Test: rank(M) == rank(M w/o col).
        Mf = np.delete(M, remove_idx, axis=1)
        return np.linalg.matrix_rank(Mf) == np.linalg.matrix_rank(M)

    best_rank = RANK
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    current_rank = RANK
    M = build_M(U, V, W, current_rank)
    target = np.ones(n * m * p)

    for attempt in range(6):
        if time.time() > _deadline - 8:
            break
        if current_rank <= best_rank:
            break
        improved = False
        for remove_idx in range(current_rank):
            if time.time() > _deadline - 5:
                break
            if can_reduce(M, target, current_rank, remove_idx):
                # candidate: column is redundant -> drop and verify exactly
                U_try = np.delete(U, remove_idx, axis=1)
                V_try = np.delete(V, remove_idx, axis=1)
                W_try = np.delete(W, remove_idx, axis=1)
                if verify(U_try, V_try, W_try, current_rank - 1):
                    best_rank = current_rank - 1
                    best_U, best_V, best_W = U_try, V_try, W_try
                    U, V, W = U_try, V_try, W_try
                    current_rank = best_rank
                    M = build_M(U, V, W, current_rank)
                    improved = True
                    break
        if not improved:
            break

    if best_rank < RANK:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    # ---- Tensor-flattening matrix and randomized nullspace reduction ----
    def build_M(U_mat, V_mat, W_mat, r):
        """M[ijk, l] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]"""
        M = np.zeros((n * m * p, r))
        for l in range(r):
            colU = U_mat[:, l]
            colV = V_mat[:, l]
            colW = W_mat[:, l]
            for i in range(n):
                base = i * m * p
                for j in range(m):
                    a = colU[i * m + j]
                    if a == 0.0:
                        continue
                    for k in range(p):
                        M[base + j * p + k, l] = a * colV[j * p + k] * colW[k * n + i]
        return M

    best_rank = RANK
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_rank = RANK

    # Phase 1: Randomized nullspace detection via random linear combination
    # Build flattening matrices F_i (each n*m x p) for each "i-slice" of the tensor.
    # A random linear combination R = sum_i alpha_i * F_i gives a single matrix.
    # The rank of R (over reals) lower-bounds the tensor rank.
    # More importantly: if rank(R) < current_rank, the nullspace of R^T gives
    # linear dependencies among the rank-1 terms, identifying removable columns.
    rng = np.random.default_rng(42)
    reduced = False

    for trial in range(12):
        if time.time() > _deadline - 10:
            break
        # Random coefficients for the p flattening slices
        alpha = rng.standard_normal(p)
        # Build combined matrix: sum_k alpha[k] * F_k where F_k[i*m+j, l] = U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l]
        # F_k has shape (n*m, r)
        F_combined = np.zeros((n * m, current_rank))
        for k in range(p):
            ak = alpha[k]
            if abs(ak) < 1e-12:
                continue
            for i in range(n):
                wk = W_mat_col = W[:, k * n + i]  # (r,)
                for j in range(m):
                    uj = U[i * m + j]  # (r,)
                    vj = V[j * p + k]  # (r,)
                    F_combined[i * m + j] += ak * (uj * vj * wk)
        # Now check: does the nullspace of F_combined^T (i.e., vectors c with F_combined^T c = 0)
        # give us a direction to eliminate?
        # Actually, we want: is there a column l that is "redundant"?
        # Better approach: compute SVD of F_combined. If rank < current_rank,
        # the right nullspace vectors indicate dependencies.
        try:
            U_svd, s_vals, Vt = np.linalg.svd(F_combined, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        tol = 1e-8 * max(s_vals) if len(s_vals) > 0 else 1e-8
        rank_F = int(np.sum(s_vals > tol))
        if rank_F < current_rank:
            # Nullspace of F_combined^T: vectors c in R^r with F_combined^T c = 0
            # These c satisfy: for all (i,j), sum_l c[l] * F_combined[i*m+j, l] = 0
            # This means the linear combination sum_l c[l] * (rank-1 term l) has zero
            # contribution to the (i,j,k)-flattened slice for the specific alpha.
            # We need to verify this more carefully: a column l can be removed if
            # removing it still reproduces the tensor. Let's just try greedy removal
            # guided by which columns have small "contribution" in the nullspace sense.
            # Simpler: try removing each column and verify.
            pass  # fall through to greedy

    # Phase 2: Greedy column elimination with verification
    M = build_M(U, V, W, current_rank)
    target = np.ones(n * m * p)
    # Check if M @ ones == target (sanity)
    # Actually target should be: M @ e_l for each l should give the l-th rank-1 tensor.
    # The decomposition is valid iff sum_l M[:,l] = target (ones vector).
    # A column l is removable iff sum_{l' != l} M[:,l'] = target, i.e., M[:,l] = 0.
    # That's too strict. The correct test: the tensor is T = sum_l u_l ⊗ v_l ⊗ w_l.
    # Removing column l gives T' = T - u_l ⊗ v_l ⊗ w_l. T' == T iff u_l ⊗ v_l ⊗ w_l = 0.
    # So a column is removable iff it's a zero column in the tensor sense.
    # But we can also do COMBINED removal: remove a set S if sum_{l in S} u_l⊗v_l⊗w_l = 0.
    # For single column: check if the rank-1 tensor is zero.
    # More useful: check if column l is a linear combination of others in a way
    # that preserves the tensor. This is the nullspace approach.

    # Let's use the direct approach: for each column, check if removing it preserves
    # the tensor identity. Use the M matrix: M @ 1 = target.
    # After removing column l: (M without col l) @ 1_{r-1} should still = target.
    # This is equivalent to: M[:,l] must be in the column space of M without col l
    # AND the sum of remaining columns must equal target.
    # Actually simplest: just verify after removal.

    for attempt in range(8):
        if time.time() > _deadline - 5:
            break
        improved = False
        for remove_idx in range(current_rank):
            if time.time() > _deadline - 3:
                break
            U_try = np.delete(U, remove_idx, axis=1)
            V_try = np.delete(V, remove_idx, axis=1)
            W_try = np.delete(W, remove_idx, axis=1)
            if verify(U_try, V_try, W_try, current_rank - 1):
                best_rank = current_rank - 1
                best_U, best_V, best_W = U_try, V_try, W_try
                U, V, W = U_try, V_try, W_try
                current_rank = best_rank
                M = build_M(U, V, W, current_rank)
                improved = True
                break
        if not improved:
            break

    # Phase 3: Try removing pairs of columns (faster than individual if they cancel)
    if current_rank > 50 and time.time() < _deadline - 8:
        for i in range(current_rank):
            if time.time() > _deadline - 4:
                break
            for j in range(i + 1, current_rank):
                if time.time() > _deadline - 3:
                    break
                mask = np.ones(current_rank, dtype=bool)
                mask[i] = False
                mask[j] = False
                U_try = U[:, mask]
                V_try = V[:, mask]
                W_try = W[:, mask]
                if verify(U_try, V_try, W_try, current_rank - 2):
                    best_rank = current_rank - 2
                    best_U, best_V, best_W = U_try, V_try, W_try
                    U, V, W = U_try, V_try, W_try
                    current_rank = best_rank
                    break
            if current_rank < best_rank:
                break

    if best_rank < RANK:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE

Wait, I referenced `W_mat_col` which is undefined. Let me fix that. Also, the random nullspace phase as written doesn't actually do anything useful (it just passes). Let me rewrite more carefully to actually use the nullspace to guide which columns to try removing first, and fix the variable name issue.

