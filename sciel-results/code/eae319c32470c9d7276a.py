"""Matrix multiplication 2x3x10 - rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # safety margin from 150s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Build the target tensor T[i, j, k] = 1 for all i,j,k
    # Shape: (n, m, p) = (2, 3, 10)
    T = np.ones((n, m, p))
    
    # Start with schoolbook decomposition (rank = n*m*p = 60)
    rank_init = n * m * p
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_init
    
    # Try to improve using numerical optimization
    # Strategy: try various ranks and use alternating least squares
    
    def validate(U_mat, V_mat, W_mat):
        """Check if decomposition is exact."""
        # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        r = U_mat.shape[1]
        T_recon = np.zeros((n, m, p))
        for c in range(r):
            u = U_mat[:, c].reshape(n, m)
            v = V_mat[:, c].reshape(m, p)
            w = W_mat[:, c].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(T_recon, T, atol=1e-10)
    
    def try_rank(r):
        """Try to find a rank-r decomposition using ALS."""
        if r >= best_rank:
            return None
        if time.time() > DEADLINE - 10:
            return None
        
        # Random initialization
        U_mat = np.random.randn(n * m, r) * 0.1
        V_mat = np.random.randn(m * p, r) * 0.1
        W_mat = np.random.randn(p * n, r) * 0.1
        
        # Alternating least squares
        for iteration in range(200):
            if time.time() > DEADLINE - 5:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (i,j), this is: T[i,j,:] = U[i*m+j,:]^T * diag stuff
                    # Actually: T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c])
                    # So T[i,j,:] = U[i*m+j,:]^T @ M where M[c,k] = V[j*p+k,c] * W[k*n+i,c]
                    M = np.zeros((r, p))
                    for c in range(r):
                        for k in range(p):
                            M[c, k] = V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    # T[i,j,:] = U[i*m+j,:]^T @ M  =>  T[i,j,:]^T = M^T @ U[i*m+j,:]
                    # Solve M^T @ U[i*m+j,:] = T[i,j,:]^T
                    try:
                        U_mat[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each (j,k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (j,k): T[:,j,k] = V[j*p+k,:]^T @ N where N[c,i] = U[i*m+j,c] * W[k*n+i,c]
                    N = np.zeros((r, n))
                    for c in range(r):
                        for i in range(n):
                            N[c, i] = U_mat[i * m + j, c] * W_mat[k * n + i, c]
                    try:
                        V_mat[j * p + k, :] = np.linalg.lstsq(N.T, T[:, j, k], rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each (k,i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # For fixed (k,i): T[i,:,k] = W[k*n+i,:]^T @ P where P[c,j] = U[i*m+j,c] * V[j*p+k,c]
                    P = np.zeros((r, m))
                    for c in range(r):
                        for j in range(m):
                            P[c, j] = U_mat[i * m + j, c] * V_mat[j * p + k, c]
                    try:
                        W_mat[k * n + i, :] = np.linalg.lstsq(P.T, T[i, :, k], rcond=None)[0]
                    except:
                        pass
        
        # Check if we found a valid decomposition
        if validate(U_mat, V_mat, W_mat):
            return (U_mat, V_mat, W_mat, r)
        return None
    
    # Try ranks from low to high
    for r in range(30, min(55, best_rank)):
        if time.time() > DEADLINE - 15:
            break
        result = try_rank(r)
        if result is not None:
            U_mat, V_mat, W_mat, new_rank = result
            best_U, best_V, best_W = U_mat, V_mat, W_mat
            best_rank = new_rank
            break
    
    # Convert to list format
    def to_list(mat):
        return [[mat[i, j] for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
