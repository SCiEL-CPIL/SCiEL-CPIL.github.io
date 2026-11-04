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

    def build_tensor_matrix(U, V, W, r):
        """Build the tensor flattening matrix M[xyz, l] = U[i,j,l]*V[j,k,l]*W[k,i,l] (vectorized)"""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        u = U.reshape(n, m, r)  # (i, j, l)
        v = V.reshape(m, p, r)  # (j, k, l)
        w = W.reshape(p, n, r)  # (k, i, l)
        # T[i,j,k,l] = u[i,j,l] * v[j,k,l] * w[k,i,l]
        # einsum: ijl,jkl,kil -> ijk l
        T = np.einsum('ijl,jkl,kil->ijkl', u, v, w)
        # Flatten: M[ijk, l]
        M = T.reshape(n * m * p, r)
        return M

    def try_reduce_rank(U, V, W, r, max_iterations=20):
        """Iteratively try to reduce rank by finding nullspace vectors."""
        best_U, best_V, best_W, best_r = U.copy(), V.copy(), W.copy(), r
        
        for iteration in range(max_iterations):
            if time.time() > _deadline - 15:
                break
            if best_r <= 1:
                break
                
            M = build_tensor_matrix(best_U, best_V, best_W, best_r)
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            except:
                break
                
            tol = 1e-10
            rank_M = np.sum(s > tol)
            
            if rank_M >= best_r:
                break  # No nullspace, can't reduce further
            
            # Get nullspace vectors
            null_space = Vt[rank_M:, :].T  # (best_r, best_r - rank_M)
            
            found_reduction = False
            for nv in null_space:
                if time.time() > _deadline - 15:
                    break
                    
                # Try eliminating each column with significant coefficient
                abs_nv = np.abs(nv)
                candidates = np.where(abs_nv > tol)[0]
                
                for idx in candidates:
                    if time.time() > _deadline - 15:
                        break
                    
                    U_new = np.delete(best_U, idx, axis=1)
                    V_new = np.delete(best_V, idx, axis=1)
                    W_new = np.delete(best_W, idx, axis=1)
                    
                    if verify(U_new, V_new, W_new, best_r - 1):
                        best_U, best_V, best_W = U_new, V_new, W_new
                        best_r -= 1
                        found_reduction = True
                        break
                
                if found_reduction:
                    break
        
        return best_U, best_V, best_W, best_r

    # Build the rank-66 decomposition
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

        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U66[A00, col] += 1; U66[A11, col] += 1
        V66[B00, col] += 1; V66[B11, col] += 1
        W66[C00, col] += 1; W66[C11, col] += 1
        col += 1

        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        U66[A10, col] += 1; U66[A11, col] += 1
        V66[B00, col] += 1
        W66[C10, col] += 1; W66[C11, col] -= 1
        col += 1

        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        U66[A00, col] += 1
        V66[B01, col] += 1; V66[B11, col] -= 1
        W66[C01, col] += 1; W66[C11, col] += 1
        col += 1

        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        U66[A11, col] += 1
        V66[B10, col] += 1; V66[B00, col] -= 1
        W66[C00, col] += 1; W66[C10, col] += 1
        col += 1

        # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
        U66[A00, col] += 1; U66[A01, col] += 1
        V66[B11, col] += 1
        W66[C00, col] -= 1; W66[C01, col] += 1
        col += 1

        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        U66[A10, col] += 1; U66[A00, col] -= 1
        V66[B00, col] += 1; V66[B01, col] += 1
        W66[C11, col] += 1
        col += 1

        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
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

        # Direct: A0*B00 -> C0, A1*B10 -> C0
        U66[A0, col] += 1; V66[B00, col] += 1; W66[C0, col] += 1
        col += 1
        U66[A1, col] += 1; V66[B10, col] += 1; W66[C0, col] += 1
        col += 1
        # Direct: A0*B01 -> C1, A1*B11 -> C1
        U66[A0, col] += 1; V66[B01, col] += 1; W66[C1, col] += 1
        col += 1
        U66[A1, col] += 1; V66[B11, col] += 1; W66[C1, col] += 1
        col += 1

    # Build the decomposition
    # Row 0: direct terms for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    # Verify rank-66
    if not verify(U66, V66, W66, 66):
        # Fallback to schoolbook
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

    # Try to reduce rank
    U_final, V_final, W_final, final_rank = try_reduce_rank(U66, V66, W66, 66)
    
    # Additional pass: try removing pairs of columns simultaneously
    for iter2 in range(5):
        if time.time() > _deadline - 15:
            break
        if final_rank <= 2:
            break
        M = build_tensor_matrix(U_final, V_final, W_final, final_rank)
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except:
            break
        tol = 1e-10
        rank_M = np.sum(s > tol)
        if rank_M >= final_rank - 1:
            break
        null_space = Vt[rank_M:, :].T
        found = False
        for nv in null_space:
            if time.time() > _deadline - 15:
                break
            abs_nv = np.abs(nv)
            candidates = np.where(abs_nv > tol)[0]
            for i1 in candidates:
                if time.time() > _deadline - 15:
                    break
                for i2 in candidates:
                    if i2 <= i1:
                        continue
                    U_new = np.delete(U_final, [i1, i2], axis=1)
                    V_new = np.delete(V_final, [i1, i2], axis=1)
                    W_new = np.delete(W_final, [i1, i2], axis=1)
                    if verify(U_new, V_new, W_new, final_rank - 2):
                        U_final, V_final, W_final = U_new, V_new, W_new
                        final_rank -= 2
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # Final verification
    if not verify(U_final, V_final, W_final, final_rank):
        # If something went wrong, fall back to rank 66
        return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
