<<<<<<< SEARCH
    def als_decompose(U, V, W, T, max_iter=200, tol=1e-12, verbose=False):
        """Alternating Least Squares for tensor decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        for it in range(max_iter):
            # Update U: for each column r, solve least squares
            # Given V[:,r] and W[:,r], we want to find U[:,r] such that
            # sum_r U[a,r] V[b,r] W[c,r] approximates T[a,b,c]
            # For fixed r, the contribution to the residual is U[:,r] * (V[:,r] outer W[:,r])
            # We can solve all r simultaneously using block structure
            
            # Build the linear system for U
            # For each (a,b,c): T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Fix V,W, solve for U
            # This is equivalent to: for each a, solve for U[a,:]
            # T[a,:,:] = sum_r U[a,r] V[:,r] W[:,r]
            # Let M_a be the matrix where M_a[b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We want M_a ≈ T[a,:,:]
            # This is a linear system in U[a,:]
            
            # More efficiently: for each a, we have a linear system
            # Let's vectorize: create a big matrix
            # For all (a,b,c), the equation is:
            # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            
            # We can write this as: for each a,
            # U[a,:] @ (V outer W) = T[a,:,:]
            # where (V outer W) has shape (R, m*p*p*n)
            
            # Actually, let's use the fact that for fixed V,W:
            # The mapping U -> tensor is linear
            # We can solve using least squares
            
            # For efficiency, solve for U by flattening
            # T_flat has shape (n*m * m*p * p*n,)
            # The linear operator maps U (flattened, shape n*m*R) to T_flat
            # This is a huge system, so let's solve per-row of U
            
            # Alternative: solve for each column r of U separately is wrong
            # because they're coupled through the sum
            
            # Let's solve the full least squares problem
            # Reshape T to (n*m, m*p * p*n)
            T_2d = T.reshape(n*m, m*p * p*n)
            
            # Build the design matrix A for U
            # A has shape (n*m * m*p * p*n, n*m * R)
            # This is too large, so let's solve per-row
            
            # For each a in range(n*m):
            # We need to find U[a,:] such that
            # sum_r U[a,r] * (V[:,r] outer W[:,r]) ≈ T[a,:,:]
            # Let M = V outer W, shape (R, m*p, p*n)
            # Then for each a: U[a,:] @ M.reshape(R, -1) = T[a].reshape(-1)
            
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, m*p * p*n)  # (R, m*p*p*n)
            
            # For each a: solve U[a,:] @ M_flat.T = T[a].reshape(-1)
            # This is: M_flat @ U[a,:] = T[a].reshape(-1)
            # Wait, let me be careful with dimensions
            
            # M_flat has shape (R, m*p*p*n)
            # We want: sum_r U[a,r] M_flat[r, :] = T[a].reshape(-1)
            # So: M_flat.T @ U[a,:] = T[a].reshape(-1)
            # M_flat.T has shape (m*p*p*n, R)
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                # Solve: M_flat.T @ U[a,:] = T[a].reshape(-1)
                # This is an underdetermined system (R unknowns, m*p*p*n equations)
                # Use least squares
                try:
                    U_new[a, :] = np.linalg.lstsq(M_flat.T, T[a].reshape(-1), rcond=None)[0]
                except:
                    U_new[a, :] = U[a, :]
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, R)
            M2_flat = M2.transpose(2, 0, 1).reshape(R, n*m * p*n)  # (R, n*m*p*n)
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                # We need: sum_r V[b,r] * (U[:,r] outer W[:,r]) ≈ T[:,b,:]
                # Let's extract T[:,b,:]
                T_slice = T[:, b, :]  # (n*m, p*n)
                # We want: M2_flat.T @ V[b,:] = T_slice.reshape(-1)
                # M2_flat.T has shape (n*m*p*n, R)
                try:
                    V_new[b, :] = np.linalg.lstsq(M2_flat.T, T_slice.reshape(-1), rcond=None)[0]
                except:
                    V_new[b, :] = V[b, :]
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
            M3_flat = M3.transpose(2, 0, 1).reshape(R, n*m * m*p)  # (R, n*m*m*p)
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                # We need: sum_r W[c,r] * (U[:,r] outer V[:,r]) ≈ T[:,:,c]
                T_slice = T[:, :, c]  # (n*m, m*p)
                try:
                    W_new[c, :] = np.linalg.lstsq(M3_flat.T, T_slice.reshape(-1), rcond=None)[0]
                except:
                    W_new[c, :] = W[c, :]
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if verbose:
                print(f"Iter {it}: err = {err:.2e}")
            
            if err < tol:
                if verbose:
                    print(f"Converged at iter {it}")
                break
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
        
        return U, V, W, err
=======
    def als_decompose(U, V, W, T, max_iter=200, tol=1e-12, verbose=False):
        """Alternating Least Squares for tensor decomposition with efficient block solves."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        for it in range(max_iter):
            # Update U: Solve for all rows of U simultaneously
            # For each a: sum_r U[a,r] * (V[:,r] outer W[:,r]) ≈ T[a,:,:]
            # M = einsum('br,cr->bcr', V, W) has shape (m*p, p*n, R)
            # For row a of U, we need: M_flat.T @ U[a,:] ≈ T[a].reshape(-1)
            # where M_flat has shape (R, m*p*p*n)
            
            M = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, R)
            M_flat = M.transpose(2, 0, 1).reshape(R, m*p * p*n)  # (R, m*p*p*n)
            M_T = M_flat.T  # (m*p*p*n, R)
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                rhs = T[a].reshape(-1)
                # Solve M_T @ U[a,:] = rhs using lstsq
                U_new[a, :] = np.linalg.lstsq(M_T, rhs, rcond=None)[0]
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)  # (n*m, p*n, R)
            M2_flat = M2.transpose(2, 0, 1).reshape(R, n*m * p*n)  # (R, n*m*p*n)
            M2_T = M2_flat.T  # (n*m*p*n, R)
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                rhs = T[:, b, :].reshape(-1)
                V_new[b, :] = np.linalg.lstsq(M2_T, rhs, rcond=None)[0]
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)  # (n*m, m*p, R)
            M3_flat = M3.transpose(2, 0, 1).reshape(R, n*m * m*p)  # (R, n*m*m*p)
            M3_T = M3_flat.T  # (n*m*m*p, R)
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                rhs = T[:, :, c].reshape(-1)
                W_new[c, :] = np.linalg.lstsq(M3_T, rhs, rcond=None)[0]
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if verbose:
                print(f"Iter {it}: err = {err:.2e}")
            
            if err < tol:
                if verbose:
                    print(f"Converged at iter {it}")
                break
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
        
        return U, V, W, err
    
    def polish(U, V, W, T, max_iter=50, tol=1e-14):
        """High-precision polish pass to achieve near-exact solution."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        prev_err = np.inf
        
        for it in range(max_iter):
            # Update U
            M = np.einsum('br,cr->bcr', V, W)
            M_flat = M.transpose(2, 0, 1).reshape(R, m*p * p*n)
            M_T = M_flat.T
            
            U_new = np.zeros_like(U)
            for a in range(n*m):
                rhs = T[a].reshape(-1)
                U_new[a, :] = np.linalg.lstsq(M_T, rhs, rcond=None)[0]
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W)
            M2_flat = M2.transpose(2, 0, 1).reshape(R, n*m * p*n)
            M2_T = M2_flat.T
            
            V_new = np.zeros_like(V)
            for b in range(m*p):
                rhs = T[:, b, :].reshape(-1)
                V_new[b, :] = np.linalg.lstsq(M2_T, rhs, rcond=None)[0]
            
            # Update W
            M3 = np.einsum('ar,br->abr', U, V)
            M3_flat = M3.transpose(2, 0, 1).reshape(R, n*m * m*p)
            M3_T = M3_flat.T
            
            W_new = np.zeros_like(W)
            for c in range(p*n):
                rhs = T[:, :, c].reshape(-1)
                W_new[c, :] = np.linalg.lstsq(M3_T, rhs, rcond=None)[0]
            
            U, V, W = U_new, V_new, W_new
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if err < tol:
                break
            if abs(prev_err - err) < 1e-16:
                break
            prev_err = err
        
        return U, V, W, err
>>>>>>> REPLACE

