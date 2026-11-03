"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    # Build a block-structured decomposition:
    # - Split columns into 3 groups of 2
    # - For each group, use Strassen for the 2x2 sub-blocks in rows 1-2
    # - For row 0, use direct terms
    # This gives rank 66. Then attempt to reduce rank by finding
    # linear dependence among the rank-1 terms via nullspace detection.

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

        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1

        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1

        # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1

        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1

        # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1

        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1

        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
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

        # Direct: A0*B00 -> C0, A1*B10 -> C0
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        # Direct: A0*B01 -> C1, A1*B11 -> C1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
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
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank by finding linear dependencies among the 66 terms
    # A tensor decomposition term c is a rank-1 tensor T_c = u_c ⊗ v_c ⊗ w_c
    # where u_c[i,j] = U[i*m+j, c], v_c[j,k] = V[j*p+k, c], w_c[k,i] = W[k*n+i, c]
    # The tensor T = sum_c T_c
    # If there's a linear dependence sum_d alpha_d T_d = 0, we can eliminate one term.
    
    # Flattening approach: for each term c, create a vector by flattening u_c ⊗ v_c ⊗ w_c
    # But that's expensive. Instead, use the fact that if sum alpha_d (u_d ⊗ v_d ⊗ w_d) = 0,
    # then for any fixed i,k: sum_d alpha_d u_d[i,:] ⊗ w_d[k,i] = 0 (as a vector in j-space)
    # More practically, check if the 66 rank-1 tensors are linearly dependent.
    
    # Use a more efficient check: the tensor T has entries T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We want to find if there's a non-trivial vector alpha such that
    # sum_c alpha_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = 0 (as a 3D tensor)
    
    # This is equivalent to: for all (i,j,k): sum_c alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 0
    # This is a system of n*m*p = 72 equations in 66 unknowns (alpha_0,...,alpha_65)
    # If the system has a non-trivial solution, we can reduce the rank.
    
    # Build the system: M[eq, c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where eq = index of (i,j,k)
    
    if time.time() > _deadline - 20:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the 72x66 matrix M where M[(i,j,k), c] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    M = np.zeros((n*m*p, 66))
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(66):
                    M[eq_idx, c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                eq_idx += 1
    
    # Find nullspace of M (vectors alpha such that M @ alpha = 0)
    try:
        svd_val, svd_vecs = np.linalg.svd(M, full_matrices=False)
        # The nullspace is spanned by right singular vectors corresponding to near-zero singular values
        tol = 1e-10 * max(svd_val) if len(svd_val) > 0 else 1e-10
        null_mask = svd_val < tol
        null_dim = np.sum(null_mask)
        
        if null_dim > 0 and time.time() < _deadline - 10:
            # Get nullspace basis vectors
            null_basis = svd_vecs[:, null_mask]
            
            # Try to eliminate one term at a time
            # For each null vector, find the term with largest |alpha_c| and try to eliminate it
            best_rank = 66
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for ni in range(null_dim):
                if time.time() > _deadline - 10:
                    break
                alpha = null_basis[:, ni]
                # Find the term with largest absolute coefficient
                max_idx = np.argmax(np.abs(alpha))
                if abs(alpha[max_idx]) < 1e-10:
                    continue
                
                # Try to eliminate term max_idx by expressing it in terms of others
                # New decomposition: for c != max_idx, keep as is
                # We need to verify that removing term max_idx and adjusting others still gives the same tensor
                # Actually, if sum_c alpha_c T_c = 0, then T_max_idx = -sum_{c != max_idx} (alpha_c / alpha_max_idx) T_c
                # So we can't just remove it; we need to combine terms.
                # 
                # Alternative: use the nullspace to find a smaller generating set.
                # If alpha is in the nullspace, then the 66 terms are linearly dependent.
                # We can drop one term if it can be expressed as a combination of others.
                # But for tensor rank, we need the remaining terms to still span the same tensor.
                # 
                # Simpler approach: just check if we can find a subset of 65 terms that still
                # generates the same tensor. This is hard in general.
                # 
                # Practical approach: since we have a null vector, we know the terms are dependent.
                # The minimal rank is at most 65. Let's try to find an explicit rank-65 decomposition.
                # 
                # Use the null vector to modify one term:
                # T'_c = T_c + alpha_c/alpha_max_idx * T_max_idx for c != max_idx
                # T'_max_idx = 0
                # This gives the same total tensor but with term max_idx zeroed out.
                
                new_U = U.copy()
                new_V = V.copy()
                new_W = W.copy()
                
                for c in range(66):
                    if c == max_idx:
                        continue
                    scale = -alpha[c] / alpha[max_idx]
                    new_U[:, c] += scale * U[:, max_idx]
                    new_V[:, c] += scale * V[:, max_idx]
                    new_W[:, c] += scale * W[:, max_idx]
                
                # Zero out term max_idx
                new_U[:, max_idx] = 0
                new_V[:, max_idx] = 0
                new_W[:, max_idx] = 0
                
                # Verify
                if verify(new_U, new_V, new_W, 66):
                    # Check if term max_idx is actually zero (it should be by construction)
                    # Remove the zero term to get rank 65
                    keep = [c for c in range(66) if c != max_idx]
                    U65 = new_U[:, keep]
                    V65 = new_V[:, keep]
                    W65 = new_W[:, keep]
                    
                    if verify(U65, V65, W65, 65):
                        best_rank = 65
                        best_U = U65
                        best_V = V65
                        best_W = W65
                        break
            else:
                # No break, try all null vectors
                pass
            
            if best_rank < 66:
                U, V, W = best_U, best_V, best_W
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
