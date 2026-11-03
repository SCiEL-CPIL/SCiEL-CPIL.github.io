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

    # Vectorized dependency check
    # Construct the matrix M where M[flat_idx, c] = U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We want to check if any column c can be expressed as a linear combination of the others.
    # This is equivalent to checking if the rank of M is less than the number of columns.
    # Or, more specifically, if we remove column r, is the rank of M[:, r] in the span of M[:, c!=r]?
    
    r0 = 66
    # Build M efficiently
    # M is 72 x 66
    # We can build it by iterating over the 66 terms
    # But 66 is small.
    
    # Let's try to find a column to remove.
    # We will try to remove each column and see if the system is consistent.
    # This is expensive if done naively, but 66 * (solve 72x65) is feasible.
    
    # Precompute the tensor vectors for each column
    # T_c is a vector of length 72
    # T_c[i*m*p + j*p + k] = U66[i*m+j, c] * V66[j*p+k, c] * W66[k*n+i, c]
    
    T_vecs = np.zeros((r0, n * m * p))
    for c in range(r0):
        # Reshape U, V, W columns to tensors
        u_c = U66[:, c].reshape(n, m)
        v_c = V66[:, c].reshape(m, p)
        w_c = W66[:, c].reshape(p, n)
        # Compute product tensor
        t_c = np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c)
        T_vecs[c, :] = t_c.flatten()
        
    # Check rank of T_vecs.T (72 x 66)
    # If rank < 66, there is a dependency.
    # We want to find a sparse dependency if possible, or just any.
    
    try:
        # Use SVD to find null space
        U_svd, s, Vh = np.linalg.svd(T_vecs.T, full_matrices=False)
        rank = np.sum(s > 1e-9)
        
        if rank < r0:
            # There are r0 - rank null space vectors
            # The null space is the last r0 - rank rows of Vh
            null_space = Vh[rank:, :].T
            # We have dependencies.
            # We want to remove one column.
            # Pick a column with a large coefficient in a null vector?
            # Or just try to remove the last column and see if we can adjust?
            # Actually, if there is a dependency, we can simply drop one column 
            # if the remaining columns still span the target tensor?
            # No, the dependency is sum alpha_c T_c = 0.
            # If we drop column r, we can't just drop it. We must replace the target?
            # No, the target is fixed.
            # If sum alpha_c T_c = 0, then sum_{c != r} (-alpha_c/alpha_r) T_c = T_r.
            # So T_r is in the span of the others.
            # But we need sum T_c = Target.
            # If we remove T_r, we lose T_r from the sum.
            # We need the remaining sum to equal Target.
            # Currently sum_{c != r} T_c = Target - T_r.
            # We need to scale the remaining terms? No, the coefficients are fixed to 1 in the sum.
            # Wait, the decomposition is sum_c T_c = Target.
            # If T_r = sum_{c != r} beta_c T_c, then
            # Target = sum_{c != r} T_c + T_r = sum_{c != r} T_c + sum_{c != r} beta_c T_c = sum_{c != r} (1 + beta_c) T_c.
            # So we can define new coefficients for the remaining terms.
            # But the interface requires U, V, W matrices.
            # The "coefficient" is embedded in the entries of U, V, W.
            # If we have a linear dependency among the *tensors* T_c, it does not mean we can simply remove a column from U, V, W.
            # We would need to find new U', V', W' for the remaining columns that produce the same sum.
            # This is a hard problem (tensor decomposition).
            
            # However, if the dependency is "structural" (e.g., two columns are identical), we can merge them?
            # No, they are rank-1 terms.
            
            # Given the complexity, and that 66 is a strong candidate, 
            # and the time limit, I will return the rank 66 solution.
            pass
    except:
        pass

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
# EVOLVE-BLOCK-END
