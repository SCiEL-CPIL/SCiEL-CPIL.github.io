import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def schoolbook(n, m, p):
    """Standard n*m*p rank decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def try_rank(rank, n, m, p, max_iter=200, tol=1e-8, seed=0, deadline=None):
    """ALS with structured initialization for matrix multiplication tensor."""
    rng = np.random.RandomState(seed)
    
    # Initialize with structured random values
    U = rng.randn(n * m, rank) * 0.3
    V = rng.randn(m * p, rank) * 0.3
    W = rng.randn(p * n, rank) * 0.3
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            return None
        
        # Update U: for each (i,j), solve sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        for i in range(n):
            for j in range(m):
                # Build system: for each k, sum_c U[i*m+j,c] * A[c,k] = 1
                # where A[c,k] = V[j*p+k,c] * W[k*n+i,c]
                A = np.zeros((rank, p))
                for c in range(rank):
                    for k in range(p):
                        A[c, k] = V[j * p + k, c] * W[k * n + i, c]
                # Want U[i*m+j,:] @ A = [1,1,...,1]
                # U[i*m+j,:] is a row vector of length rank
                # Solve: A.T @ U[i*m+j,:].T = ones(p)
                try:
                    sol = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Update V: for each (j,k), sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1
        for j in range(m):
            for k in range(p):
                A = np.zeros((rank, n))
                for c in range(rank):
                    for i in range(n):
                        A[c, i] = U[i * m + j, c] * W[k * n + i, c]
                try:
                    sol = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Update W: for each (k,i), sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
        for k in range(p):
            for i in range(n):
                A = np.zeros((rank, m))
                for c in range(rank):
                    for j in range(m):
                        A[c, j] = U[i * m + j, c] * V[j * p + k, c]
                try:
                    sol = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    error += (s - 1.0) ** 2
        
        if error < tol:
            return U, V, W, error
    
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    U, V, W = schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try to find lower rank decompositions
    for rank in range(rank_school - 1, 10, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(30):
            if time.time() > deadline:
                break
            
            result = try_rank(rank, n, m, p, max_iter=300, tol=1e-10, seed=seed, deadline=deadline)
            
            if result is not None:
                U_np, V_np, W_np, error = result
                if error < 1e-6:
                    U_list = U_np.tolist()
                    V_list = V_np.tolist()
                    W_list = W_np.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        best_U, best_V, best_W = U_list, V_list, W_list
                        best_rank = rank
                        found = True
                        break
        
        if found:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
