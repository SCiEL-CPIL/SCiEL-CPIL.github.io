import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if A[i,j]B[j,k] contributes to C[i,k]
    # Actually T[i,j,k] is the coefficient.
    # The tensor is 1 for all i,j,k? No.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has T[i,j,k] = 1 for all i,j,k?
    # Yes, because the multiplication is linear and each term A[i,j]B[j,k] appears exactly once in C[i,k].
    # So T is the all-ones tensor of size 4x4x9.
    
    T = np.ones((n, m, p))
    
    # We want U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 140 # 10s margin
    
    # Try ranks from 40 to 143
    for r in range(40, 144):
        if time.time() > deadline:
            break
            
        # Try a few restarts for each rank
        found = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize V and W randomly
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                # For fixed i,j, T[i,j,:] = sum_c U[i,j,c] * (V[:,k,c] * W[:,i,c])? No.
                # Let's reshape.
                # T is (n, m, p).
                # We can flatten T to (n*m, p).
                # T_flat[i*m+j, k] = sum_c U[i*m+j, c] * (sum_k? No)
                
                # Let's construct the linear system for U.
                # For each c, the term is U[i,j,c] * V[j,k,c] * W[k,i,c].
                # This is not a simple matrix product because of the index coupling.
                # However, for a fixed c, let's define a matrix M_c of size (n*m, p).
                # M_c[i*m+j, k] = V[j*p+k, c] * W[k*n+i, c].
                # Then T_flat = sum_c U[:, c] * M_c? No.
                # T_flat[i*m+j, k] = sum_c U[i*m+j, c] * M_c[i*m+j, k].
                # This is a rank-1 update in the "c" dimension?
                # Actually, T_flat = U @ M, where M is a 3D tensor?
                # Let M be a tensor of shape (r, n*m, p).
                # M[c, i*m+j, k] = V[j*p+k, c] * W[k*n+i, c].
                # Then T_flat = sum_c U[:, c] * M[c, :, :].
                # This is equivalent to:
                # T_flat = (U * M).sum(axis=0)? No.
                # T_flat[i,j,k] = sum_c U[i,j,c] M[c,i,j,k].
                # We can reshape this to a matrix multiplication.
                # Let T_mat be T_flat reshaped to (n*m, p).
                # Let U_mat be U (n*m, r).
                # Let M_mat be M reshaped to (r, n*m * p).
                # This doesn't work directly because M depends on i,j,k.
                
                # We can solve for U by treating it as a linear system.
                # Variables: U[i*m+j, c] for all i,j,c. Total n*m*r variables.
                # Equations: T[i,j,k] for all i,j,k. Total n*m*p equations.
                # Coefficient for U[i0*m+j0, c0] in equation (i,j,k) is:
                # 1 if i0==i, j0==j, and then V[j*k, c0] * W[k*i, c0].
                # So the coefficient is V[j*p+k, c0] * W[k*n+i, c0] if i0==i, j0==j, else 0.
                
                # We can build the design matrix A for U.
                # A has shape (n*m*p, n*m*r).
                # A[e, u] where e = i*m*p + j*p + k, u = i*m + j, c.
                # A[e, u] = V[j*p+k, c] * W[k*n+i, c] if (i,j) from e matches (i,j) from u.
                
                # This matrix is sparse.
                # We can use numpy to build it or solve iteratively.
                
                # Given the small size (144 equations, 16*r variables), we can build the dense matrix.
                A = np.zeros((n*m*p, n*m*r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = i*m*p + j*p + k
                            for c in range(r):
                                u = i*m + j, c # index in U
                                # A[e, u] = V[j*p+k, c] * W[k*n+i, c]
                                # Note: u is (i*m+j, c)
                                A[e, (i*m+j)*r + c] = V[j*p+k, c] * W[k*n+i, c]
                
                # Solve U = argmin ||A U_vec - T_vec||^2
                T_vec = T.flatten()
                U_vec, residuals, rank, sv = np.linalg.lstsq(A, T_vec, rcond=None)
                U = U_vec.reshape(n*m, r)
                
                # Similarly for V
                # Variables: V[j*p+k, c]. Total m*p*r.
                # Equations: T[i,j,k].
                # Coefficient for V[j0*p+k0, c0] in T[i,j,k] is:
                # 1 if j0==j, k0==k, then U[i*m+j, c0] * W[k*n+i, c0].
                B = np.zeros((n*m*p, m*p*r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = i*m*p + j*p + k
                            for c in range(r):
                                v = j*p + k, c
                                B[e, (j*p+k)*r + c] = U[i*m+j, c] * W[k*n+i, c]
                V_vec, residuals, rank, sv = np.linalg.lstsq(B, T_vec, rcond=None)
                V = V_vec.reshape(m*p, r)
                
                # Similarly for W
                C = np.zeros((n*m*p, p*n*r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = i*m*p + j*p + k
                            for c in range(r):
                                w = k*n + i, c
                                C[e, (k*n+i)*r + c] = U[i*m+j, c] * V[j*p+k, c]
                W_vec, residuals, rank, sv = np.linalg.lstsq(C, T_vec, rcond=None)
                W = W_vec.reshape(p*n, r)
                
                # Check error
                # Reconstruct T
                T_rec = np.zeros((n, m, p))
                for c in range(r):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_rec[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                                
                err = np.linalg.norm(T_rec - T)
                if err < 1e-9:
                    # Check if entries are close to half-integers
                    # Multiply by 2
                    U2 = U * 2
                    V2 = V * 2
                    W2 = W * 2
                    if np.allclose(U2, np.round(U2), atol=1e-6) and \
                       np.allclose(V2, np.round(V2), atol=1e-6) and \
                       np.allclose(W2, np.round(W2), atol=1e-6):
                        U = np.round(U2) / 2
                        V = np.round(V2) / 2
                        W = np.round(W2) / 2
                        # Verify exactness
                        T_rec2 = np.zeros((n, m, p))
                        for c in range(r):
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        T_rec2[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if np.allclose(T_rec2, T, atol=1e-9):
                            best_rank = r
                            best_U = U
                            best_V = V
                            best_W = W
                            found = True
                            break
            if found:
                break
        
        if found:
            break
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
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
