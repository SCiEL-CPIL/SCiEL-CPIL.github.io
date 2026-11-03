"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: find rank-minimal decomposition
    # Schoolbook rank is n*m*p = 84
    # We'll try to find a lower rank
    
    deadline = time.time() + 110  # 10s safety margin
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # Compute the tensor: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)  
        # W shape: (p*n, rank)
        
        # Reshape for efficient computation
        # For each r, we have outer product of U[:,r], V[:,r], W[:,r]
        # Sum over r
        
        # Efficient way: use einsum
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's compute the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # This is a 3-way contraction
        
        # Use einsum: 'ijr,jkr,kr->ijk' but wait, W is indexed by (k,i) so W_reshaped[k,i,r]
        # So: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        T = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # T should be all ones
        max_error = np.max(np.abs(T - 1.0))
        return max_error < tol
    
    def try_rank(r, max_iterations=500, seed=42):
        """Try to find a decomposition of rank r."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() > deadline - 20:
                break
            
            # Alternate: fix V, W, optimize U; fix U, W, optimize V; fix U, V, optimize W
            # For each (i,j), we need sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
            # This is: U[i*m+j,:] * (elementwise product of V[j*p+k,:] and W[k*n+i,:] for each k)
            # Actually for fixed i,j, we have p equations (one for each k):
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for k=0,...,p-1
            # This is a linear system in U[i*m+j,:]: A_{k,r} = V[j*p+k,r] * W[k*n+i,r]
            # Then A @ U[i*m+j,:] = ones(p)
            
            # Similarly for V and W
            
            # Optimize U
            for i in range(n):
                for j in range(m):
                    # Build A: p x r
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ u = ones(p)
                    try:
                        u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass
            
            # Optimize V
            for j in range(m):
                for k in range(p):
                    # Build A: n x r
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    # Solve A @ v = ones(n)
                    try:
                        v, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass
            
            # Optimize W
            for k in range(p):
                for i in range(n):
                    # Build A: m x r
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve A @ w = ones(m)
                    try:
                        w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass
            
            # Check error
            if it % 10 == 0:
                if check_decomposition(U, V, W, tol=1e-4):
                    # Found a valid decomposition
                    return U, V, W, 0.0
                else:
                    # Compute current error
                    U_reshaped = U.reshape(n, m, r)
                    V_reshaped = V.reshape(m, p, r)
                    W_reshaped = W.reshape(p, n, r)
                    T = np.einsum('ijr,jkr,kr->ijk', U_reshaped, V_reshaped, W_reshaped)
                    error = np.max(np.abs(T - 1.0))
                    if error < best_error:
                        best_error = error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        return best_U, best_V, best_W, best_error
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) = 28
    # But realistically, for 3x4x7, the rank is probably around 50-70
    
    # Let's try a few candidate ranks
    candidate_ranks = [84, 80, 76, 72, 68, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28]
    
    best_result = None
    best_rank = 84
    
    # First, get the schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_result = (U_sb, V_sb, W_sb, 84)
    
    # Try to find lower ranks
    for r in candidate_ranks:
        if r >= 84:
            continue
        if time.time() > deadline - 30:
            break
        
        U, V, W, error = try_rank(r, max_iterations=200, seed=r)
        
        if error < 1e-4:
            # Found a valid decomposition
            best_result = (U, V, W, r)
            best_rank = r
            break  # Found the minimum, stop
    
    # Convert to lists
    U, V, W, rank = best_result
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

