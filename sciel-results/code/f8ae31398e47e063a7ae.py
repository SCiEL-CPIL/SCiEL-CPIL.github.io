import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _try_gradient_descent(n, m, p, rank, max_iters=5000, seed=42, lr_init=0.01):
    """
    Try to find a rank-`rank` decomposition via gradient descent on the
    factor matrices. We minimize sum of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Normalize to make products order 1
    scale = (rank / (n * m * p)) ** (1/3)
    U = U / scale
    V = V / scale
    W = W / scale
    
    lr = lr_init
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if it % 500 == 499:
            lr *= 0.7
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        err = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # s = dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    grad = s - 1.0
                    err += grad * grad
                    dU[i * m + j] += grad * V[j * p + k] * W[k * n + i]
                    dV[j * p + k] += grad * U[i * m + j] * W[k * n + i]
                    dW[k * n + i] += grad * U[i * m + j] * V[j * p + k]
        
        rms_err = np.sqrt(err / (n * m * p))
        
        if rms_err < best_err:
            best_err = rms_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if rms_err < 1e-9:
            break
        
        # Gradient step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    if best_err < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def _try_small_integer_search(n, m, p, rank, max_trials=5000, seed=42):
    """
    Try to find a rank-`rank` decomposition with entries from a small set.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Try with entries in {-1, 0, 1}
    vals = [-1, 0, 1]
    
    for t in range(max_trials):
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_mixed_precision_search(n, m, p, rank, max_trials=2000, seed=42):
    """
    Try entries from a mixed set including half-integers.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Entries from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for t in range(max_trials):
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_structured_binary(n, m, p, rank, seed=42):
    """
    Try a structured approach using binary representations of indices.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For each column, create a pattern based on binary digits
    for c in range(rank):
        # Use the column index to determine which bits to use
        for b in range(8):
            if (c >> b) & 1:
                # Add contribution from bit b
                # U: based on bits of i, j
                for i in range(n):
                    for j in range(m):
                        bit_i = (i >> (b % 3)) & 1
                        bit_j = (j >> (b % 4)) & 1
                        if (bit_i ^ bit_j) == (c % 2):
                            U[i * m + j, c] += 1
                        else:
                            U[i * m + j, c] -= 1
                
                # V: based on bits of j, k
                for j in range(m):
                    for k in range(p):
                        bit_j = (j >> (b % 4)) & 1
                        bit_k = (k >> (b % 3)) & 1
                        if (bit_j ^ bit_k) == ((c + 1) % 2):
                            V[j * p + k, c] += 1
                        else:
                            V[j * p + k, c] -= 1
                
                # W: based on bits of k, i
                for k in range(p):
                    for i in range(n):
                        bit_k = (k >> (b % 3)) & 1
                        bit_i = (i >> (b % 3)) & 1
                        if (bit_k ^ bit_i) == ((c + 2) % 2):
                            W[k * n + i, c] += 1
                        else:
                            W[k * n + i, c] -= 1
    
    # Normalize
    norm = np.sqrt(rank)
    U = U / norm
    V = V / norm
    W = W / norm
    
    # Verify
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try various ranks from 80 down to 50
    for rank in range(80, 49, -1):
        if time.time() > deadline - 20:
            break
        
        found = None
        
        # First try small integer search (fast)
        for seed in range(3):
            if time.time() > deadline - 20:
                break
            result = _try_small_integer_search(n, m, p, rank, max_trials=1000, seed=seed)
            if result is not None:
                found = result
                break
        
        # Try mixed precision search
        if found is None:
            for seed in range(2):
                if time.time() > deadline - 20:
                    break
                result = _try_mixed_precision_search(n, m, p, rank, max_trials=500, seed=seed)
                if result is not None:
                    found = result
                    break
        
        # Try gradient descent with multiple seeds
        if found is None:
            for seed in range(5):
                if time.time() > deadline - 20:
                    break
                result = _try_gradient_descent(n, m, p, rank, max_iters=3000, seed=seed)
                if result is not None:
                    U, V, W = result
                    if _verify(U, V, W, n, m, p):
                        found = (U, V, W)
                        break
        
        # Try structured binary approach
        if found is None:
            for seed in range(3):
                if time.time() > deadline - 20:
                    break
                result = _try_structured_binary(n, m, p, rank, seed=seed)
                if result is not None:
                    found = result
                    break
        
        if found is not None:
            U, V, W = found
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best = {"U": U, "V": V, "W": W}
                break
    
    return best
# EVOLVE-BLOCK-END
