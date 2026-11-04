"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 4, 6
    nmp = n * m * p  # 72

    def build_rank66():
        """Build the rank-66 block-structured decomposition."""
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

            U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
            col += 1
            U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
            col += 1

        # Row 0: direct terms
        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)

        # Rows 1-2: Strassen
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        return np.array(U66, dtype=float), np.array(V66, dtype=float), np.array(W66, dtype=float)

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_tensor_matrix(U_mat, V_mat, W_mat, r):
        """Build the 72 x r matrix where M[xyz, l] = U[xm+y, l] * V[yp+z, l] * W[zp+x, l]."""
        M = np.zeros((nmp, r))
        for l in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m * p + j * p + k, l] = (
                            U_mat[i * m + j, l] *
                            V_mat[j * p + k, l] *
                            W_mat[k * n + i, l]
                        )
        return M

    # Build rank-66 decomposition
    U, V, W = build_rank66()

    # Verify
    if not verify(U, V, W, 66):
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

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 66

    # Iterative nullspace elimination to reduce rank
    # Each iteration: build tensor matrix, find nullspace, try to eliminate columns
    for round_num in range(10):
        if time.time() > _deadline - 5:
            break

        r_cur = best_rank
        U_cur = best_U
        V_cur = best_V
        W_cur = best_W

        # Build tensor matrix
        M = build_tensor_matrix(U_cur, V_cur, W_cur, r_cur)

        # SVD to find nullspace
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10
            rank_M = int(np.sum(s > tol))

            if rank_M >= r_cur:
                break  # No more nullspace directions

            # Get nullspace vectors
            null_dim = r_cur - rank_M
            null_space = Vt[rank_M:, :].T  # (r_cur, null_dim)

            # Try each nullspace vector to find the best column to eliminate
            found = False
            for nv in null_space:
                if time.time() > _deadline - 5:
                    break

                # Find the column with the largest absolute coefficient
                abs_nv = np.abs(nv)
                max_idx = int(np.argmax(abs_nv))
                if abs_nv[max_idx] < tol:
                    continue

                # Try eliminating this column
                U_new = np.delete(U_cur, max_idx, axis=1)
                V_new = np.delete(V_cur, max_idx, axis=1)
                W_new = np.delete(W_cur, max_idx, axis=1)

                if verify(U_new, V_new, W_new, r_cur - 1):
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    best_rank = r_cur - 1
                    found = True
                    break

            if not found:
                # Try all columns individually as a fallback
                for idx in range(r_cur):
                    if time.time() > _deadline - 5:
                        break
                    U_new = np.delete(U_cur, idx, axis=1)
                    V_new = np.delete(V_cur, idx, axis=1)
                    W_new = np.delete(W_cur, idx, axis=1)
                    if verify(U_new, V_new, W_new, r_cur - 1):
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                        best_rank = r_cur - 1
                        found = True
                        break

            if not found:
                break  # Can't reduce further

        except:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
