"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    import scipy.optimize
    import scipy.optimize

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
        """Build the tensor flattening matrix M[xyz, l] = U[i,j,l]*V[j,k,l]*W[k,i,l]"""
        M = np.zeros((n * m * p, r))
        for l in range(r):
            u_l = U[:, l].reshape(n, m)
            v_l = V[:, l].reshape(m, p)
            w_l = W[:, l].reshape(p, n)
            # T_l[i,j,k] = u_l[i,j] * v_l[j,k] * w_l[k,i]
            T_l = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_l[i, j, k] = u_l[i, j] * v_l[j, k] * w_l[k, i]
            M[:, l] = T_l.flatten()
        return M

    def try_reduce_rank(U, V, W, r, max_iterations=10):
        """Iteratively try to reduce rank by finding nullspace vectors."""
        best_U, best_V, best_W, best_r = U.copy(), V.copy(), W.copy(), r
        
        for iteration in range(max_iterations):
            if time.time() > _deadline - 10:
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
                if time.time() > _deadline - 10:
                    break
                    
                # Try eliminating each column with significant coefficient
                abs_nv = np.abs(nv)
                candidates = np.where(abs_nv > tol)[0]
                
                for idx in candidates:
                    if time.time() > _deadline - 10:
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

    # Try to reduce rank using local search / optimization
    best_U, best_V, best_W, best_rank = U66.copy(), V66.copy(), W66.copy(), 66
    
    # Function to flatten U, V, W for optimization
    def flatten_factors(U, V, W, r):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def unflatten_factors(x, r):
        size_U = n * m * r
        size_V = m * p * r
        size_W = p * n * r
        U = x[:size_U].reshape(n * m, r)
        V = x[size_U:size_U+size_V].reshape(m * p, r)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(p * n, r)
        return U, V, W

    def objective(x, r):
        U, V, W = unflatten_factors(x, r)
        u = U.reshape(n, m, r)
        v = V.reshape(m, p, r)
        w = W.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return (T_reconstructed - target).flatten()

    # Try to drop each column and optimize
    for col_to_drop in range(66):
        if time.time() > _deadline - 20:
            break
            
        # Create initial guess by dropping the column
        U_init = np.delete(U66, col_to_drop, axis=1)
        V_init = np.delete(V66, col_to_drop, axis=1)
        W_init = np.delete(W66, col_to_drop, axis=1)
        
        r_new = 65
        x0 = flatten_factors(U_init, V_init, W_init, r_new)
        
        try:
            # Use scipy.optimize.least_squares
            result = scipy.optimize.least_squares(objective, x0, args=(r_new,), method='lm', max_nfev=500)
            
            if result.success or np.min(result.cost) < 1e-6:
                # Verify the solution
                U_opt, V_opt, W_opt = unflatten_factors(result.x, r_new)
                if verify(U_opt, V_opt, W_opt, r_new):
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, r_new
                    break # Found rank 65, stop searching for lower (unlikely to be 64 quickly)
        except Exception:
            continue

    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # If something went wrong, fall back to rank 66
        return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
