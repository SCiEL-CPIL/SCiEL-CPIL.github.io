import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # Try ranks from low to high
    for rank in range(24, 64):
        for attempt in range(20):
            np.random.seed(attempt * 1000 + rank)
            
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            converged = False
            for iteration in range(500):
                # Solve for U: for each (i,j), solve linear system
                # For each (i,j), we have p equations:
                # Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0..p-1
                # This is: (V[j*p:j*p+p, :] * W[0:n*n:..., :] element-wise) @ U[i*m+j, :].T = ones(p)
                # More precisely: let M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then M_ij @ U[i*m+j, :].T = ones(p)
                
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build M_ij: shape (p, rank)
                        M_ij = np.zeros((p, rank))
                        for k in range(p):
                            M_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve M_ij @ u = ones(p)
                        try:
                            u, residuals, rank_val, sv = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                            new_U[i*m+j, :] = u
                        except:
                            break
                        if iteration == 0:
                            pass
                    else:
                        continue
                    break
                else:
                    # Update V
                    new_V = np.zeros_like(V)
                    for j in range(m):
                        for k in range(p):
                            # For fixed (j,k), equations over i:
                            # Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for i=0..n-1
                            # M_jk[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            M_jk = np.zeros((n, rank))
                            for i in range(n):
                                M_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            try:
                                v, _, _, _ = np.linalg.lstsq(M_jk, np.ones(n), rcond=None)
                                new_V[j*p+k, :] = v
                            except:
                                break
                            if iteration == 0:
                                pass
                        else:
                            continue
                        break
                    else:
                        # Update W
                        new_W = np.zeros_like(W)
                        for k in range(p):
                            for i in range(n):
                                # For fixed (k,i), equations over j:
                                # Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for j=0..m-1
                                # M_ki[j, r] = U[i*m+j, r] * V[j*p+k, r]
                                M_ki = np.zeros((m, rank))
                                for j in range(m):
                                    M_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                                try:
                                    w, _, _, _ = np.linalg.lstsq(M_ki, np.ones(m), rcond=None)
                                    new_W[k*n+i, :] = w
                                except:
                                    break
                                if iteration == 0:
                                    pass
                            else:
                                continue
                            break
                        else:
                            # Check convergence
                            old_U, old_V, old_W = U, V, W
                            U, V, W = new_U, new_V, new_W
                            
                            # Check residual
                            max_err = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                        err = abs(val - 1.0)
                                        if err > max_err:
                                            max_err = err
                            
                            if max_err < 1e-8:
                                converged = True
                                break
                            if max_err > 1.0:
                                break  # Diverging
                if converged:
                    break
                if max_err > 1.0:
                    break
            
            if converged:
                # Verify
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-6:
                    # Convert to lists
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
