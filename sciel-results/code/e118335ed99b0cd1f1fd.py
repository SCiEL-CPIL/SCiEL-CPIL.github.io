"""Matrix multiplication 4x4 by 4x5 rank-minimal decomposition search."""
import time
import random
import itertools

DIMENSIONS = [4, 4, 5]

def _build_decomp(n, m, p, R, U, V, W):
    """Build the decomposition dictionary from matrices."""
    return {"U": [list(U[i]) for i in range(n*m)],
            "V": [list(V[j]) for j in range(m*p)],
            "W": [list(W[k]) for k in range(p*n)]}

def _compute_tensor(n, m, p, R, U, V, W):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                T[(i,j,k)] = s
    return T

def _verify(n, m, p, R, U, V, W):
    """Check if the decomposition is exact (all tensor entries = 1)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return _build_decomp(n, m, p, rank, U, V, W)

def _random_init(n, m, p, R):
    """Initialize U, V, W with random values."""
    U = [[random.gauss(0, 1) for _ in range(R)] for _ in range(n*m)]
    V = [[random.gauss(0, 1) for _ in range(R)] for _ in range(m*p)]
    W = [[random.gauss(0, 1) for _ in range(R)] for _ in range(p*n)]
    return U, V, W

def _compute_residuals(n, m, p, R, U, V, W):
    """Compute residuals T[i,j,k] - 1 for all (i,j,k)."""
    res = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(R):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res[(i,j,k)] = s - 1
    return res

def _gradient_step(n, m, p, R, U, V, W, lr):
    """One gradient descent step. Returns new U, V, W."""
    # Compute residuals
    res = _compute_residuals(n, m, p, R, U, V, W)
    
    U_new = [[0.0]*R for _ in range(n*m)]
    V_new = [[0.0]*R for _ in range(m*p)]
    W_new = [[0.0]*R for _ in range(p*n)]
    
    for c in range(R):
        for i in range(n):
            for j in range(m):
                # dU[i*m+j][c] = sum_k res[(i,j,k)] * V[j*p+k][c] * W[k*n+i][c]
                grad = 0.0
                for k in range(p):
                    grad += res[(i,j,k)] * V[j*p+k][c] * W[k*n+i][c]
                U_new[i*m+j][c] = U[i*m+j][c] - lr * grad
        
        for j in range(m):
            for k in range(p):
                # dV[j*p+k][c] = sum_i res[(i,j,k)] * U[i*m+j][c] * W[k*n+i][c]
                grad = 0.0
                for i in range(n):
                    grad += res[(i,j,k)] * U[i*m+j][c] * W[k*n+i][c]
                V_new[j*p+k][c] = V[j*p+k][c] - lr * grad
        
        for k in range(p):
            for i in range(n):
                # dW[k*n+i][c] = sum_j res[(i,j,k)] * U[i*m+j][c] * V[j*p+k][c]
                grad = 0.0
                for j in range(m):
                    grad += res[(i,j,k)] * U[i*m+j][c] * V[j*p+k][c]
                W_new[k*n+i][c] = W[k*n+i][c] - lr * grad
    
    return U_new, V_new, W_new

def _search_low_rank(n, m, p, target_rank, time_budget):
    """Try to find a decomposition of the given rank via gradient descent."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_result = None
    best_error = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        U, V, W = _random_init(n, m, p, target_rank)
        
        lr = 0.001
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            
            U, V, W = _gradient_step(n, m, p, target_rank, U, V, W, lr)
            
            if step % 100 == 0:
                # Check error
                err = max(abs(v) for v in _compute_residuals(n, m, p, target_rank, U, V, W).values())
                if err < best_error:
                    best_error = err
                    best_result = (U[:], V[:], W[:])
                
                if err < 1e-6:
                    # Verify and return
                    if _verify(n, m, p, target_rank, U, V, W):
                        return _build_decomp(n, m, p, target_rank, U, V, W)
                
                # Adaptive learning rate
                if err < 1e-3:
                    lr = 0.0001
                elif err < 0.1:
                    lr = 0.0005
                else:
                    lr = 0.001
        
        # Final check
        if best_result and best_error < 1e-9:
            U, V, W = best_result
            if _verify(n, m, p, target_rank, U, V, W):
                return _build_decomp(n, m, p, target_rank, U, V, W)
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    
    # Known lower bound: rank >= 49 for 4x4 by 4x5
    # Try to find rank 49 decomposition
    start_time = time.time()
    time_budget = 100  # Leave 20s safety margin
    
    result = _search_low_rank(n, m, p, 49, time_budget)
    if result is not None:
        return result
    
    # If 49 failed, try 50
    if time.time() - start_time < 100:
        result = _search_low_rank(n, m, p, 50, 30)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook
