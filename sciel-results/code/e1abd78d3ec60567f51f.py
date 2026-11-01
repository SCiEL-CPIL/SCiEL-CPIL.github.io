import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 3]

def verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check if U, V, W define the tensor
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i, j, k] should be 1 for all i, j, k in the multiplication sum.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def als_decompose(rank, n, m, p, max_iter=500, tol=1e-10, seed=0):
    """Use ALS to find rank-r decomposition of matrix multiplication tensor."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Target tensor: T[i, j, k] = 1 for all valid i, j, k
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    for iteration in range(max_iter):
        # Update U
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    # Compute the required value
                    target = 1.0
                    # Sum over other components
                    s = 0.0
                    for k in range(p):
                        s += V[j * p + k][c] * W[k * n + i][c]
                    if abs(s) > 1e-12:
                        U[i * m + j][c] = target / s
                    else:
                        U[i * m + j][c] = 0
        
        # Update V
        for c in range(rank):
            for j in range(m):
                for k in range(p):
                    target = 1.0
                    s = 0.0
                    for i in range(n):
                        s += U[i * m + j][c] * W[k * n + i][c]
                    if abs(s) > 1e-12:
                        V[j * p + k][c] = target / s
                    else:
                        V[j * p + k][c] = 0
        
        # Update W
        for c in range(rank):
            for k in range(p):
                for i in range(n):
                    target = 1.0
                    s = 0.0
                    for j in range(m):
                        s += U[i * m + j][c] * V[j * p + k][c]
                    if abs(s) > 1e-12:
                        W[k * n + i][c] = target / s
                    else:
                        W[k * n + i][c] = 0
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    error += (s - 1.0) ** 2
        
        if error < tol:
            break
    
    return U, V, W, error

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Start with schoolbook solution
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    start_time = time.time()
    deadline = 120.0
    
    # Try to find lower rank decompositions
    for rank in range(rank_school - 1, 10, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(20):
            if time.time() - start_time > deadline - 5:
                break
            
            U_np, V_np, W_np, error = als_decompose(rank, n, m, p, max_iter=200, tol=1e-8, seed=seed)
            
            if error < 1e-6:
                # Convert to lists
                U_list = U_np.tolist()
                V_list = V_np.tolist()
                W_list = W_np.tolist()
                
                if verify(U_list, V_list, W_list):
                    best_U, best_V, best_W = U_list, V_list, W_list
                    best_rank = rank
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, continue to lower ranks
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
