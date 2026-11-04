import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
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
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find lower rank using random restarts and optimization
    for R in range(best_rank - 1, 55, -1):
        if time.time() > deadline - 5:
            break
        
        for seed in range(500):
            if time.time() > deadline - 2:
                break
            
            np.random.seed(seed * 104729 + R * 31)
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Try to optimize
            U, V, W, success = try_optimize(U, V, W, n, m, p, deadline)
            if success:
                best_U, best_V, best_W, best_rank = U, V, W, R
                break
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def try_optimize(U, V, W, n, m, p, deadline):
    """Try to optimize U, V, W to satisfy constraints."""
    # Use a simple iterative method
    lr = 0.1
    for it in range(5000):
        if time.time() > deadline - 1:
            break
        
        # Compute gradient numerically (approximate)
        # For simplicity, use a small step and check if objective decreases
        obj_before = compute_objective(U, V, W, n, m, p)
        if obj_before < 1e-12:
            return U, V, W, True
        
        # Random perturbation (simple stochastic gradient)
        U_new = U + lr * np.random.randn(*U.shape)
        V_new = V + lr * np.random.randn(*V.shape)
        W_new = W + lr * np.random.randn(*W.shape)
        
        obj_after = compute_objective(U_new, V_new, W_new, n, m, p)
        
        if obj_after < obj_before:
            U, V, W = U_new, V_new, W_new
        else:
            lr *= 0.9
    
    obj_final = compute_objective(U, V, W, n, m, p)
    return U, V, W, obj_final < 1e-8

def compute_objective(U, V, W, n, m, p):
    """Compute sum of squares of constraint violations."""
    total = 0.0
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                for k in range(p):
                    val = np.dot(W[k * n + i], U[i * m + j1] * V[j2 * p + k])
                    target = 1.0 if j1 == j2 else 0.0
                    total += (val - target) ** 2
    return total
