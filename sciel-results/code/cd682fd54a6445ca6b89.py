import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', W[:, r], U[:, r], V[:, r])
    
    return np.allclose(T_decomp, T, atol=1e-6)

def _schoolbook(n=3, m=3, p=6):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _gradient_descent(rank, n=3, m=3, p=6, max_iters=30000, seed=42, lr=0.01,
                      init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Gradient descent to find a rank-r decomposition with adaptive learning rate.
    """
    rng = np.random.RandomState(seed)
    
    if init_U is not None:
        U = init_U.copy()
    else:
        U = rng.randn(n * m, rank) * 0.1
    
    if init_V is not None:
        V = init_V.copy()
    else:
        V = rng.randn(m * p, rank) * 0.1
    
    if init_W is not None:
        W = init_W.copy()
    else:
        W = rng.randn(p * n, rank) * 0.1
    
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        T_decomp = np.einsum('ar,br,cr->abc', W, U, V)
        R = T_decomp - T
        
        grad_U = np.einsum('abc,cr,ar->br', R, V, W)
        grad_V = np.einsum('abc,br,ar->cr', R, U, W)
        grad_W = np.einsum('abc,br,cr->ar', R, U, V)
        
        err = np.linalg.norm(R)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-8:
            return best_U, best_V, best_W
        
        grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
        if grad_norm > 1e-12:
            step_lr = lr * err / (grad_norm + 1e-8)
        else:
            step_lr = lr
        
        U -= step_lr * grad_U
        V -= step_lr * grad_V
        W -= step_lr * grad_W
        
        if it % 1000 == 0 and it > 0 and err > 1e-4:
            noise_scale = max(0.005, 0.05 * err)
            U += noise_scale * rng.randn(n * m, rank)
            V += noise_scale * rng.randn(m * p, rank)
            W += noise_scale * rng.randn(p * n, rank)
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_result = (U_fb, V_fb, W_fb, schoolbook_rank)
    
    ranks_to_try = list(range(24, schoolbook_rank))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = None
        
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            if time.time() > deadline:
                break
            
            result = _gradient_descent(rank, n, m, p, max_iters=5000, seed=seed, deadline=deadline)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W, rank)
                    break
        
        if found is not None:
            best_result = found
            break
    
    U, V, W, rank = best_result
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
