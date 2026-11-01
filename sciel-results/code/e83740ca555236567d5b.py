<<<<<<< SEARCH
    # Verify the tensor decomposition
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True
=======
    # Verify the tensor decomposition
    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        # T[i, j, jp, k] = sum_r U[i, j, r] V[jp, k, r] W[k, i, r]
        T = np.einsum('ijr,jkr,kir->ijpk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target: 1 if j == jp, else 0
        D = np.zeros((n, m, m, p), dtype=int)
        for i in range(n):
            for k in range(p):
                D[i, :, :, k] = np.eye(m)
                
        if not np.allclose(T, D, atol=tol):
            return False
        return True
>>>>>>> REPLACE
