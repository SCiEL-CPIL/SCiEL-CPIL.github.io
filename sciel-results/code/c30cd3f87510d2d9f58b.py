import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m + j][t] * V[j*p + k][t] * W[k*n + i][t]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _objective(U, V, W, n, m, p, rank):
    """Compute sum of squared errors for all tensor entries."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
                err += (s - 1.0) ** 2
    return err

def _try_optimize(U0, V0, W0, n, m, p, rank, max_iter=2000, lr=0.01):
    """Iteratively optimize U, V, W to minimize tensor error."""
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    best_err = _objective(U, V, W, n, m, p, rank)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m + j, t] * V[j*p + k, t] * W[k*n + i, t]
                    err = s - 1.0
                    for t in range(rank):
                        dU[i*m + j, t] += 2 * err * V[j*p + k, t] * W[k*n + i, t]
                        dV[j*p + k, t] += 2 * err * U[i*m + j, t] * W[k*n + i, t]
                        dW[k*n + i, t] += 2 * err * U[i*m + j, t] * V[j*p + k, t]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        err = _objective(U, V, W, n, m, p, rank)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
    
    return best_U, best_V, best_W, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition
    np.random.seed(42)
    
    # Try ranks from 99 down to 50
    for target_rank in range(best_rank - 1, max(best_rank - 50, 0), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U_try = np.random.randn(n * m, target_rank) * 0.3
            V_try = np.random.randn(m * p, target_rank) * 0.3
            W_try = np.random.randn(p * n, target_rank) * 0.3
            
            # Optimize
            U_opt, V_opt, W_opt, err = _try_optimize(
                U_try, V_try, W_try, n, m, p, target_rank, 
                max_iter=1500, lr=0.005
            )
            
            if err < 1e-10:
                # Verify
                if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, target_rank):
                    best_rank = target_rank
                    best_U = U_opt.tolist()
                    best_V = V_opt.tolist()
                    best_W = W_opt.tolist()
                    found = True
                    break
        
        if found:
            continue
        else:
            # If we can't find rank target_rank, try lower ranks
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
