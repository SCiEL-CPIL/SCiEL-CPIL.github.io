"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 135
    n, m, p = DIMENSIONS  # 3, 4, 6

    U66 = [[0.0] * 66 for _ in range(n * m)]
    V66 = [[0.0] * 66 for _ in range(m * p)]
    W66 = [[0.0] * 66 for _ in range(p * n)]

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

        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
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

        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
        col += 1

    for t0 in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, t0, 0, t0)
        add_1x2_2x2(0, 2, 2, t0, 0, t0)

    for t0 in [0, 2, 4]:
        add_strassen(1, 0, 0, t0, 1, t0)
        add_strassen(1, 2, 2, t0, 1, t0)

    U = np.array(U66, dtype=float)
    V = np.array(V66, dtype=float)
    W = np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    if not verify(U, V, W, 66):
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col2] = 1
                    V[j * p + k, col2] = 1
                    W[k * n + i, col2] = 1
                    col2 += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: Nullspace-based column elimination
    M = np.zeros((n * m * p, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, l] = U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]

    eliminated = set()
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        null_dim = np.sum(s < tol)

        if null_dim > 0:
            nullspace = Vt[66 - null_dim:, :]

            for null_vec in nullspace:
                if time.time() > _deadline - 20:
                    break
                candidates = [i for i in range(66) if i not in eliminated]
                if not candidates:
                    break
                max_idx = max(candidates, key=lambda i: abs(null_vec[i]))
                if abs(null_vec[max_idx]) < 1e-10:
                    continue

                remaining = [i for i in range(66) if i != max_idx]
                U_test = U[:, remaining]
                V_test = V[:, remaining]
                W_test = W[:, remaining]

                if verify(U_test, V_test, W_test, len(remaining)):
                    eliminated.add(max_idx)
    except:
        pass

    if eliminated:
        remaining = [i for i in range(66) if i not in eliminated]
        U = U[:, remaining]
        V = V[:, remaining]
        W = W[:, remaining]

    current_rank = U.shape[1]

    # Phase 2: Iterative drop search
    if time.time() < _deadline - 10:
        improved = True
        while improved and time.time() < _deadline - 10:
            improved = False
            r = U.shape[1]
            for l in range(r):
                if time.time() > _deadline - 10:
                    break
                U_test = np.delete(U, l, axis=1)
                V_test = np.delete(V, l, axis=1)
                W_test = np.delete(W, l, axis=1)
                if verify(U_test, V_test, W_test, r - 1):
                    U, V, W = U_test, V_test, W_test
                    improved = True
                    break

    # Phase 3: Pairwise merge search - try to replace two columns with one
    # A merge is valid if there exists a column (u_new, v_new, w_new) such that
    # the sum of the two rank-1 tensors equals the new rank-1 tensor plus zero contribution elsewhere.
    # This is equivalent to: can we find alpha, beta such that for all (i,j,k):
    #   U[i,j,l1]*V[j,k,l1]*W[i,k,l1] + U[i,j,l2]*V[j,k,l2]*W[i,k,l2]
    #   = U_new[i,j]*V_new[j,k]*W_new[i,k]
    # We search over small integer/half-integer values for the new column.
    if time.time() < _deadline - 5:
        r = U.shape[1]
        # Try pairs of columns to see if they can be merged
        # For a merge to work, the sum of two rank-1 tensors must be rank 1.
        # T_sum[i,j,k] = U1[i,j]*V1[j,k]*W1[i,k] + U2[i,j]*V2[j,k]*W2[i,k]
        # This is rank 1 iff T_sum can be written as U_new[i,j]*V_new[j,k]*W_new[i,k]
        # We check this by computing the 3-tensor and seeing if it's rank 1.
        
        found_merge = True
        while found_merge and time.time() < _deadline - 5:
            found_merge = False
            r = U.shape[1]
            for l1 in range(r):
                if time.time() > _deadline - 5:
                    break
                for l2 in range(l1 + 1, r):
                    if time.time() > _deadline - 5:
                        break
                    # Compute the sum tensor of columns l1 and l2
                    # T_sum[i,j,k] = U[:,l1][i,j]*V[:,l1][j,k]*W[:,l1][i,k] + U[:,l2][i,j]*V[:,l2][j,k]*W[:,l2][i,k]
                    # Check if T_sum is rank 1 by checking if all 2x2 minors of appropriate unfoldings vanish
                    # More practically: try to find U_new, V_new, W_new with small entries
                    
                    # For rank-1 check: T_sum must satisfy T_sum[i1,j,k1]*T_sum[i2,j,k2] = T_sum[i1,j,k2]*T_sum[i2,j,k1] for all i1!=i2, k1!=k2, j
                    # And similarly for other mode pairs.
                    
                    # Simpler approach: just try all small integer combinations for the new column
                    # and check if removing l1, l2 and adding the new column still verifies.
                    
                    # This is expensive, so only try a limited search
                    if r > 60:  # Only do this if we haven't reduced much
                        continue
                    
                    # Try small integer values for the new column entries
                    # The new column must satisfy: for all (i,j,k)
                    # U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i] = T_sum[i,j,k]
                    # where T_sum is the sum of the two removed columns' contributions
                    
                    # We can determine U_new, V_new, W_new up to scaling from T_sum if it's rank 1.
                    # Let's check rank-1 first using a heuristic:
                    # Pick a fixed j0, then T_sum[:,j0,:] should be outer product of U_new[:,j0] and W_new[:,j0]
                    # i.e., T_sum[i,j0,k] = U_new[i*m+j0] * W_new[k*n+i] for all i,k
                    # This means T_sum[:,j0,:] has rank 1 as a matrix.
                    
                    j0 = 0
                    T_slice = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            T_slice[i, k] = (U[i*m+j0, l1]*V[j0*p+k, l1]*W[k*n+i, l1] +
                                             U[i*m+j0, l2]*V[j0*p+k, l2]*W[k*n+i, l2])
                    
                    # Check if T_slice is rank 1
                    if np.abs(T_slice).max() < 1e-15:
                        continue
                    sv = np.linalg.svd(T_slice, compute_uv=False)
                    if len(sv) < 2 or sv[1] > 1e-10 * max(sv[0], 1e-15):
                        continue
                    
                    # Also check for j1 = 1
                    j1 = 1
                    T_slice1 = np.zeros((n, p))
                    for i in range(n):
                        for k in range(p):
                            T_slice1[i, k] = (U[i*m+j1, l1]*V[j1*p+k, l1]*W[k*n+i, l1] +
                                              U[i*m+j1, l2]*V[j1*p+k, l2]*W[k*n+i, l2])
                    
                    if np.abs(T_slice1).max() < 1e-15:
                        continue
                    sv1 = np.linalg.svd(T_slice1, compute_uv=False)
                    if len(sv1) < 2 or sv1[1] > 1e-10 * max(sv1[0], 1e-15):
                        continue
                    
                    # Check consistency: the ratio U_new[i*m+j1]/U_new[i*m+j0] should equal V_new[j1*p+k]/V_new[j0*p+k] * ... 
                    # Actually, let's just try to construct the new column and verify.
                    
                    # From T_slice (j=j0): T_slice[i,k] = U_new[i*m+j0] * W_new[k*n+i]
                    # From T_slice1 (j=j1): T_slice1[i,k] = U_new[i*m+j1] * W_new[k*n+i]
                    # So T_slice1[i,k] / T_slice[i,k] = U_new[i*m+j1] / U_new[i*m+j0] (when T_slice[i,k] != 0)
                    
                    # Let's pick a reference (i0, k0) where T_slice[i0,k0] != 0
                    ref_found = False
                    for i0 in range(n):
                        for k0 in range(p):
                            if abs(T_slice[i0, k0]) > 1e-15:
                                ref_found = True
                                break
                        if ref_found:
                            break
                    
                    if not ref_found:
                        continue
                    
                    # U_new[i*m+j0] = T_slice[i,k] / W_new[k*n+i]
                    # W_new[k*n+i] is free up to scaling. Let's set W_new[k0*n+i0] = 1.
                    # Then U_new[i0*m+j0] = T_slice[i0,k0]
                    # And U_new[i*m+j0] = T_slice[i,k] * T_slice[i0,k0] / T_slice[i0,k] (when T_slice[i0,k] != 0)
                    # This is getting complex. Let's just do a direct verification search.
                    
                    # Try: set the new column to be proportional to column l1 (or l2)
                    # Actually, the simplest merge: if column l2 is a scalar multiple of l1 in all three matrices,
                    # we can just drop one. But that's already handled by the drop search.
                    
                    # Let's skip the complex merge and rely on the drop search.
                    pass

    if not verify(U, V, W, U.shape[1]):
        # Fallback to the original 66
        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
