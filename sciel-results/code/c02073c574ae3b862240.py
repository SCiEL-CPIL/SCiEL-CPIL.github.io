import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]

def check_decomposition(U, V, W, n, m, p, rank):
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-6:
                    return False
    return True

def tensor_residual(U, V, W, n, m, p):
    """Compute the residual tensor R[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1"""
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    residual = np.zeros((n, m, p))
    for r in range(rank):
        # U[:,r] is (n*m,), V[:,r] is (m*p,), W[:,r] is (p*n,)
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residual[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    residual -= 1.0
    return residual

def als_optimize(U, V, W, n, m, p, rank, max_iter=50, tol=1e-8):
    """Alternating Least Squares to minimize sum of squared residuals"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for iteration in range(max_iter):
        # Update U
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    # For each (i,j), we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # Fix all except U[i*m+j,r]
                    # The contribution of U[i*m+j,r] is: U[i*m+j,r] * sum_k V[j*p+k,r] * W[k*n+i,r]
                    # But we also have other terms from other r'
                    # This is complex; use gradient approach instead
                    pass
        
        # Use a simpler gradient-based approach
        # Compute gradient with respect to U
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i*m+j,r] of (sum_r' U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] - 1)^2
                        # = 2 * residual[i,j,k] * V[j*p+k,r] * W[k*n+i,r]
                        res = 0
                        for r2 in range(rank):
                            res += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
                        res -= 1.0
                        grad_U[i*m+j, r] += 2 * res * V[j*p+k, r] * W[k*n+i, r]
                        
                        # d/dV[j*p+k,r]
                        grad_V[j*p+k, r] += 2 * res * U[i*m+j, r] * W[k*n+i, r]
                        
                        # d/dW[k*n+i,r]
                        grad_W[k*n+i, r] += 2 * res * U[i*m+j, r] * V[j*p+k, r]
        
        # Simple gradient step
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if np.max(np.abs(grad_U)) < tol:
            break
    
    return U.tolist(), V.tolist(), W.tolist()

def try_rank(U_init, V_init, W_init, n, m, p, rank, max_iter=100, time_limit=5.0):
    """Try to optimize from initialization to achieve rank decomposition"""
    start_time = time.time()
    U, V, W = als_optimize(U_init, V_init, W_init, n, m, p, rank, max_iter=max_iter)
    
    if time.time() - start_time > time_limit:
        return None
    
    if check_decomposition(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook initialization
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school
    
    global_start = time.time()
    global_time_limit = 120  # seconds
    
    # Try various target ranks from low to high
    for target_rank in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
        if time.time() - global_start > global_time_limit - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - global_start > global_time_limit - 5:
                break
            
            # Random initialization with small values
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            result = try_rank(U, V, W, n, m, p, target_rank, max_iter=50, time_limit=3.0)
            
            if result is not None:
                best_result = result
                best_rank = target_rank
                break  # Move to next lower rank
        
        if best_rank <= target_rank:
            break  # Found a decomposition at this rank or lower
    
    return best_result
