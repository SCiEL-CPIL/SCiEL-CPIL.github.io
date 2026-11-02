"""Program for matrix multiplication 3x3x5 tensor rank minimization."""
import numpy as np
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Build the full multiplication tensor
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This represents: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition is: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    T = np.ones((n, m, p), dtype=np.float64)
    
    def verify(U, V, W, r):
        """Verify that the decomposition exactly reproduces the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def schoolbook():
        """Return the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W
    
    def try_rank_als(r, seed=42, max_iter=3000):
        """
        Try to find an exact rank-r decomposition using alternating least squares.
        Returns (U, V, W) if successful, None otherwise.
        """
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Normalize columns for stability
        for t in range(r):
            nu = np.linalg.norm(U[:, t])
            nv = np.linalg.norm(V[:, t])
            nw = np.linalg.norm(W[:, t])
            if nu > 1e-12:
                U[:, t] /= nu
            if nv > 1e-12:
                V[:, t] /= nv
            if nw > 1e-12:
                W[:, t] /= nw
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iter):
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j): T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                    # Let C[k,t] = V[j*p+k,t] * W[k*n+i,t]
                    # Then T[i,j,:] = U[i*m+j,:] @ C.T  =>  C @ U[i*m+j,:].T = T[i,j,:].T
                    C = np.zeros((p, r))
                    for t in range(r):
                        C[:, t] = V[j * p + np.arange(p), t] * W[np.arange(p) * n + i, t]
                    # Solve C @ x = T[i,j,:] where x = U[i*m+j,:]
                    try:
                        x, _, _, _ = np.linalg.lstsq(C, T[i, j, :], rcond=None)
                        new_U[i * m + j, :] = x
                    except:
                        new_U[i * m + j, :] = U[i * m + j, :]
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k): T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                    # Let D[i,t] = U[i*m+j,t] * W[k*n+i,t]
                    # Then T[:,j,k] = V[j*p+k,:] @ D.T  =>  D @ V[j*p+k,:].T = T[:,j,k].T
                    D = np.zeros((n, r))
                    for i in range(n):
                        for t in range(r):
                            D[i, t] = U[i * m + j, t] * W[k * n + i, t]
                    try:
                        x, _, _, _ = np.linalg.lstsq(D, T[:, j, k], rcond=None)
                        new_V[j * p + k, :] = x
                    except:
                        new_V[j * p + k, :] = V[j * p + k, :]
            V = new_V
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i): T[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
                    # Let E[j,t] = U[i*m+j,t] * V[j*p+k,t]
                    # Then T[i,:,k] = W[k*n+i,:] @ E.T  =>  E @ W[k*n+i,:].T = T[i,:,k].T
                    E = np.zeros((m, r))
                    for j in range(m):
                        for t in range(r):
                            E[j, t] = U[i * m + j, t] * V[j * p + k, t]
                    try:
                        x, _, _, _ = np.linalg.lstsq(E, T[i, :, k], rcond=None)
                        new_W[k * n + i, :] = x
                    except:
                        new_W[k * n + i, :] = W[k * n + i, :]
            W = new_W
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(r):
                            s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                        err += (1.0 - s) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
        
        if best_err < 1e-10 and verify(best_U, best_V, best_W, r):
            return best_U, best_V, best_W
        return None
    
    # Try ranks from a reasonable lower bound up to schoolbook rank
    # Lower bound for 3x3x5 matrix multiplication is known to be at least 15
    # (strassen rank for 2x2 is 7, so 3x3x3 is at least 19 by known results)
    # For 3x3x5, try from 15 upward
    
    for r in range(15, rank_school + 1):
        # Try multiple seeds for each rank
        for seed in range(10):
            result = try_rank_als(r, seed=42 + r * 100 + seed, max_iter=2000)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
