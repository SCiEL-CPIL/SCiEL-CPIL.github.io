import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iters=200, tol=1e-10, deadline=None):
    """
    Alternating Least Squares for rank-rank tensor decomposition.
    U: (n*m) x rank
    V: (m*p) x rank  
    W: (p*n) x rank
    Constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    # Initialize with random values
    random.seed(42)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    best_residual = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve for U[i*m+j, :] given V and W
        for i in range(n):
            for j in range(m):
                # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                # A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                # Then A @ U[i*m+j, :] = ones(p)
                A = np.zeros((rank, p))
                for k in range(p):
                    for r in range(rank):
                        A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                # Solve A @ x = ones(p)
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :] given U and W
        for j in range(m):
            for k in range(p):
                # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # A[r, i] = U[i*m+j, r] * W[k*n+i, r]
                # Then A @ V[j*p+k, :] = ones(n)
                A = np.zeros((rank, n))
                for i in range(n):
                    for r in range(rank):
                        A[r, i] = U[i*m+j, r] * W[k*n+i, r]
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :] given U and V
        for k in range(p):
            for i in range(n):
                # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # A[r, j] = U[i*m+j, r] * V[j*p+k, r]
                # Then A @ W[k*n+i, :] = ones(m)
                A = np.zeros((rank, m))
                for j in range(m):
                    for r in range(rank):
                        A[r, j] = U[i*m+j, r] * V[j*p+k, r]
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if residual < tol:
            break
    
    # Check if we found a valid solution
    if best_residual < 1e-8:
        # Convert to list of lists with rounded values
        U_list = [[round(float(x), 6) for x in row] for row in best_U]
        V_list = [[round(float(x), 6) for x in row] for row in best_V]
        W_list = [[round(float(x), 6) for x in row] for row in best_W]
        
        # Verify
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s buffer for safety
    
    # Schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Try rank 29 using ALS
    if time.time() < deadline:
        result = _als_decompose(n, m, p, rank=29, max_iters=500, tol=1e-12, deadline=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try rank 28
    if time.time() < deadline:
        result = _als_decompose(n, m, p, rank=28, max_iters=500, tol=1e-12, deadline=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
