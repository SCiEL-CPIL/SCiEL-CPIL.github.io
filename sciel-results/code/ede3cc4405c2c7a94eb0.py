"""Rewritten program for 3x3 x 3x8 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    expected = 1.0 if j == k else 0.0
                    actual = np.dot(U_arr[i*m+j, :], V_arr[j*p+l, :] * W_arr[k*n+i, :])
                    if abs(actual - expected) > 1e-6:
                        return False
    return True

def _als_decompose(R, time_budget, seed=42):
    """ALS-based tensor decomposition for the matmul tensor."""
    n, m, p = DIMENSIONS
    start = time.time()
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    for iteration in range(10000):
        if time.time() - start > time_budget:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((m*p, R))
                b_vec = np.zeros(m*p)
                idx = 0
                for k in range(m):
                    for l in range(p):
                        A_mat[idx, :] = V[j*p+l, :] * W[k*n+i, :]
                        b_vec[idx] = 1.0 if j == k else 0.0
                        idx += 1
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for l in range(p):
                A_mat = np.zeros((n*m, R))
                b_vec = np.zeros(n*m)
                idx = 0
                for i in range(n):
                    for k in range(m):
                        A_mat[idx, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_vec[idx] = 1.0 if j == k else 0.0
                        idx += 1
                try:
                    V[j*p+l, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(m):
            for i in range(n):
                A_mat = np.zeros((m*p, R))
                b_vec = np.zeros(m*p)
                idx = 0
                for jj in range(m):
                    for l in range(p):
                        A_mat[idx, :] = U[i*m+jj, :] * V[jj*p+l, :]
                        b_vec[idx] = 1.0 if jj == k else 0.0
                        idx += 1
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Check error every 10 iterations
        if iteration % 10 == 0:
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(m):
                        for l in range(p):
                            expected = 1.0 if j == k else 0.0
                            actual = np.dot(U[i*m+j, :], V[j*p+l, :] * W[k*n+i, :])
                            error += (actual - expected) ** 2
            
            if error < 1e-12:
                break
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    expected = 1.0 if j == k else 0.0
                    actual = np.dot(U[i*m+j, :], V[j*p+l, :] * W[k*n+i, :])
                    error += (actual - expected) ** 2
    
    if error < 1e-10:
        U_list = [[U[i, r] for r in range(R)] for i in range(n*m)]
        V_list = [[V[i, r] for r in range(R)] for i in range(m*p)]
        W_list = [[W[i, r] for r in range(R)] for i in range(p*n)]
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try different ranks with multiple random seeds
    for R in range(22, 40):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Try a few seeds for this rank
        for seed in range(3):
            if time.time() > deadline:
                break
            time_budget = min(remaining - 2, 12)
            result = _als_decompose(R, time_budget, seed=seed + R * 100)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
