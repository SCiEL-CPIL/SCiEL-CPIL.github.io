"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_rank66():
        U66 = np.zeros((n * m, 66))
        V66 = np.zeros((m * p, 66))
        W66 = np.zeros((p * n, 66))
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

            U66[A00, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A11, col] += 1
            V66[B00, col] += 1
            W66[C10, col] += 1; W66[C11, col] -= 1
            col += 1

            U66[A00, col] += 1
            V66[B01, col] += 1; V66[B11, col] -= 1
            W66[C01, col] += 1; W66[C11, col] += 1
            col += 1

            U66[A11, col] += 1
            V66[B10, col] += 1; V66[B00, col] -= 1
            W66[C00, col] += 1; W66[C10, col] += 1
            col += 1

            U66[A00, col] += 1; U66[A01, col] += 1
            V66[B11, col] += 1
            W66[C00, col] -= 1; W66[C01, col] += 1
            col += 1

            U66[A10, col] += 1; U66[A00, col] -= 1
            V66[B00, col] += 1; V66[B01, col] += 1
            W66[C11, col] += 1
            col += 1

            U66[A01, col] += 1; U66[A11, col] -= 1
            V66[B10, col] += 1; V66[B11, col] += 1
            W66[C00, col] += 1
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

            U66[A0, col] += 1; V66[B00, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B10, col] += 1; W66[C0, col] += 1
            col += 1
            U66[A0, col] += 1; V66[B01, col] += 1; W66[C1, col] += 1
            col += 1
            U66[A1, col] += 1; V66[B11, col] += 1; W66[C1, col] += 1
            col += 1

        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        return U66, V66, W66

    U66, V66, W66 = build_rank66()

    if not verify(U66, V66, W66, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the 72x66 matrix where entry (ijk, c) = U66[i*m+j, c] * V66[j*p+k, c] * W66[k*n+i, c]
    r0 = 66
    M = np.zeros((n * m * p, r0))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                flat_idx = i * m * p + j * p + k
                for c in range(r0):
                    M[flat_idx, c] = U66[i * m + j, c] * V66[j * p + k, c] * W66[k * n + i, c]

    # Check the rank of M and find null space for rank reduction
    try:
        rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    except:
        rank_M = 66

    best_result = {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

    if rank_M < 66:
        # Find the null space to identify linear dependencies
        try:
            null_space = np.linalg.null_space(M)
            if null_space.shape[1] > 0:
                # For each null vector, try to use it to eliminate one rank-1 term
                for ns_idx in range(null_space.shape[1]):
                    if time.time() > _deadline - 10:
                        break
                    alpha = null_space[:, ns_idx]
                    # Find the column with the largest absolute coefficient to eliminate
                    col_to_remove = np.argmax(np.abs(alpha))
                    if abs(alpha[col_to_remove]) < 1e-10:
                        continue
                    
                    # Create new matrices with rank 65
                    U65 = np.zeros((n * m, 65))
                    V65 = np.zeros((m * p, 65))
                    W65 = np.zeros((p * n, 65))
                    
                    # Copy all columns except the one to remove
                    for c in range(66):
                        if c == col_to_remove:
                            continue
                        new_c = c if c < col_to_remove else c - 1
                        # Scale by -alpha[col_to_remove] / alpha[c] if needed, but first try simple elimination
                        U65[:, new_c] = U66[:, c]
                        V65[:, new_c] = V66[:, c]
                        W65[:, new_c] = W66[:, c]
                    
                    # Now we need to add the contribution of the removed column
                    # The removed column c_r has entries U66[:,c_r], V66[:,c_r], W66[:,c_r]
                    # We need to find coefficients beta_c such that:
                    # U66[:,c_r] = sum_c beta_c * U66[:,c] for c != c_r
                    # This is a linear system, but we can use the null vector directly
                    
                    # Actually, the null vector tells us: sum_c alpha_c * M[:,c] = 0
                    # This means the tensor contribution of column c_r can be redistributed
                    # Let's try a different approach: solve for the redistribution
                    
                    # For each tensor element (i,j,k), we need:
                    # U66[i,j,c_r] * V66[j,k,c_r] * W66[k,i,c_r] = 
                    #   sum_{c != c_r} delta_c(i,j,k) * U66[i,j,c] * V66[j,k,c] * W66[k,i,c]
                    # where delta_c(i,j,k) = -alpha[c_r]/alpha[c]
                    
                    # This doesn't work directly because the delta depends on (i,j,k)
                    # Instead, we need to modify the U, V, W matrices themselves
                    
                    # Let's try: for each c != c_r, scale column c by (1 + alpha[c_r]/alpha[c])
                    # This won't work either because it's not linear in that way
                    
                    # Correct approach: We need to find new U', V', W' such that
                    # sum_{c=1}^{65} U'[i,j,c] V'[j,k,c] W'[k,i,c] = original sum
                    # This is a non-trivial problem. Let's use a simpler heuristic:
                    # Try to find if one column is a linear combination of others in each matrix
                    
                    # Check if U66[:,c_r] is in the span of U66[:,c] for c != c_r
                    U_other = np.delete(U66, col_to_remove, axis=1)
                    V_other = np.delete(V66, col_to_remove, axis=1)
                    W_other = np.delete(W66, col_to_remove, axis=1)
                    
                    # Try to solve U66[:,c_r] = U_other @ x
                    try:
                        x, residuals, rank_u, sv = np.linalg.lstsq(U_other, U66[:, col_to_remove], rcond=1e-9)
                        if residuals is not None and len(residuals) > 0 and residuals[0] < 1e-6:
                            # Check V and W similarly
                            y, residuals_v, rank_v, sv_v = np.linalg.lstsq(V_other, V66[:, col_to_remove], rcond=1e-9)
                            if residuals_v is not None and len(residuals_v) > 0 and residuals_v[0] < 1e-6:
                                z, residuals_w, rank_w, sv_w = np.linalg.lstsq(W_other, W66[:, col_to_remove], rcond=1e-9)
                                if residuals_w is not None and len(residuals_w) > 0 and residuals_w[0] < 1e-6:
                                    # Great! We can eliminate this column
                                    U65 = U_other.copy()
                                    V65 = V_other.copy()
                                    W65 = W_other.copy()
                                    if verify(U65, V65, W65, 65):
                                        best_result = {"U": U65.tolist(), "V": V65.tolist(), "W": W65.tolist()}
                                        break
                    except:
                        pass
        except:
            pass

    return best_result
# EVOLVE-BLOCK-END
