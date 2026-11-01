import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    # Target tensor T[i,j,k] = 1 for all valid indices
    T = np.ones((n, m, p))
    
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        # result[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    def als_decompose(target_rank, max_iters=300, tol=1e-10, seed=None):
        """Alternating Least Squares for tensor decomposition."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
            
        # Initialize with small random values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        prev_err = float('inf')
        
        for iter_ in range(max_iters):
            if time.time() - start_time > deadline:
                return None, None, None, None
                
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # For fixed (i,j): T[i,j,:] = U[i*m+j,:] @ (V[j,:] * W[:,i]).T
                    # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    # Solve: A.T @ U[i*m+j, :].T = T[i,j,:].T
                    # i.e., U[i*m+j, :] = lstsq(A.T, T[i,j,:])
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, T[i, j, :].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # For fixed (j,k): T[:,j,k] = V[j*p+k,:] @ (U[:,j,:] * W[k*n,:]).T
                    # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    # Solve: B.T @ V[j*p+k, :].T = T[:,j,k].T
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B.T, T[:, j, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                    # For fixed (k,i): T[i,:,k] = W[k*n+i,:] @ (U[i,:] * V[:,k]).T
                    # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve: C.T @ W[k*n+i, :].T = T[i,:,k].T
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C.T, T[i, :, k].reshape(-1), rcond=None)[0]
                    except:
                        pass
                        
            # Check convergence
            err = compute_error(U, V, W, target_rank)
            if err < tol:
                return U, V, W, err
            if abs(err - prev_err) < 1e-12:
                if err < 1e-6:
                    return U, V, W, err
                break
            prev_err = err
            
        err = compute_error(U, V, W, target_rank)
        return U, V, W, err
    
    # Try ranks from 14 upward (known minimal rank for 2x3x3 is 14)
    for target_rank in range(14, R_school + 1):
        if time.time() - start_time > deadline:
            break
            
        found = False
        max_restarts = 50 if target_rank <= 15 else 20
        
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                break
                
            U, V, W, err = als_decompose(target_rank, max_iters=200, tol=1e-9, seed=restart)
            
            if U is None:
                break
                
            if err < 1e-8:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
                
        if found:
            continue
        else:
            continue
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
