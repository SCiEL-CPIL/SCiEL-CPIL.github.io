def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 110

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, trials=5):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for _ in range(trials):
            A = np.random.randn(2, 3)
            B = np.random.randn(3, 8)
            C_expected = A @ B
            C_computed = np.zeros((2, 8))
            for c in range(U_np.shape[1]):
                u_col = U_np[:, c]
                v_col = V_np[:, c]
                w_col = W_np[:, c]
                # Reconstruct A_part and B_part? No, just compute contribution.
                # Contribution is sum over i,j,k of u[i,j] v[j,k] w[k,i] * A[i,j] B[j,k]?
                # No, the decomposition is C = sum_r (U_r * A) (V_r * B) W_r?
                # The standard form is C_{i,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} is wrong.
                # It is C = sum_r (A \circ U_r) (B \circ V_r) \circ W_r?
                # Let's check the verify function in the current code.
                # It does:
                # U_mat = U_np[:, c].reshape(2, 3)
                # V_mat = V_np[:, c].reshape(3, 8)
                # W_vec = W_np[:, c]
                # prod = U_mat @ V_mat
                # C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
                # This implies: C = sum_r (A * U_r) (B * V_r) * W_r (elementwise)?
                # No, U_mat is not A * U_r. It is just the mask.
                # The verify function assumes U, V, W are masks for A, B, C?
                # If U has 1s, it selects parts of A?
                # No, the verify function computes `prod = U_mat @ V_mat`.
                # This implies the algorithm computes products of sub-matrices defined by U and V.
                # Then it scales by W.
                # This is a specific form of decomposition.
                # Standard tensor rank: C_{i,k} = sum_r A_{i,j} B_{j,k} ...
                # The verify function is correct for the "Strassen-like" form where
                # we compute linear combinations of A and B, multiply them, and combine results.
                # U_r is the matrix of coefficients for A.
                # V_r is the matrix of coefficients for B.
                # W_r is the vector of coefficients for C.
                # C = sum_r (A U_r^T? No)
                # Let's trace:
                # U_mat is 2x3. V_mat is 3x8.
                # prod = U_mat @ V_mat.
                # This is NOT A @ B.
                # Unless U_mat is a mask that extracts A?
                # If U_mat is 0/1 mask, then `U_mat @ V_mat` is not related to A.
                # The verify function in the seed code is WRONG for general tensor decomposition.
                # It assumes U and V are the actual A and B? No, they are fixed matrices.
                # If U and V are fixed, `U_mat @ V_mat` is a fixed matrix.
                # Then `C_computed` is a fixed linear combination of that fixed matrix.
                # It does NOT depend on A and B!
                # The verify function is broken.
                # It uses `A = np.random.randn(2, 3)` but never uses A in the computation of C_computed.
                # It computes `prod = U_mat @ V_mat` which is constant.
                # So it checks if a constant matrix equals `A @ B` which is random.
                # This will always fail.
                # This explains why the current code falls back to schoolbook.
