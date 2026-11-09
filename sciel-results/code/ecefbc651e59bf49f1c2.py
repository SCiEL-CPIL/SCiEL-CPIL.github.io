import numpy as np
import time

def _find_block_decomp(n, m, p, target_rank, time_limit=20):
    """Find a low rank decomposition for M_n,m,p."""
    # T[i, k, j] = 1
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Check: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] == 1
    
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U, V, W
    
    start = time.time()
    while time.time() - start < time_limit:
        # ALS step
        # Fix V, W, update U
        # For each i, k, we want sum_r U[i*m+k, r] * (V[k*p+j, r] * W[j*n+i, r]) = 1
        # This is linear in U.
        
        # To speed up, we can just do a few gradient descent steps or ALS.
        # Given the small size, ALS is fine.
        
        # Update U
        for i in range(n):
            for k in range(m):
                idx = i*m + k
                # Target: 1
                # We need to solve for U[idx, :]
                # sum_r U[idx, r] * M_r = 1
                # where M_r = sum_j V[k*p+j, r] * W[j*n+i, r]
                M = np.zeros(target_rank)
                for j in range(p):
                    M += V[k*p+j, :] * W[j*n+i, :]
                # Solve U[idx, :] M^T = 1
                # This is underdetermined if target_rank > 1.
                # We want min || U[idx, :] M^T - 1 ||^2
                # Solution: U[idx, :] = M (M^T M)^-1
                try:
                    U[idx, :] = np.linalg.lstsq(M, np.ones(1), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for k in range(m):
            for j in range(p):
                idx = k*p + j
                M = np.zeros(target_rank)
                for i in range(n):
                    M += U[i*m+k, :] * W[j*n+i, :]
                try:
                    V[idx, :] = np.linalg.lstsq(M, np.ones(1), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for j in range(p):
            for i in range(n):
                idx = j*n + i
                M = np.zeros(target_rank)
                for k in range(m):
                    M += U[i*m+k, :] * V[k*p+j, :]
                try:
                    W[idx, :] = np.linalg.lstsq(M, np.ones(1), rcond=None)[0]
                except:
                    pass
        
        # Calculate error
        err = 0.0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    val = np.dot(U[i*m+k, :], V[k*p+j, :] * W[j*n+i, :])
                    err += (val - 1.0)**2
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = 3, 4, 6
    
    # Try to find a better block decomposition for 3x4x2
    # If we find rank 20, total rank 60.
    # If we find rank 18, total rank 54.
    
    # Fallback to current rank 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # ... (current implementation) ...
