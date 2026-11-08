"""Matrix multiplication 2,4,6 — rank-minimal exact tensor decomposition."""

import numpy as np
import time

DIMENSIONS = [2, 4, 6]


# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, T, tol=1e-6):
    """Check if U, V, W (each shape (dim, rank)) exactly reconstruct T."""
    # T_hat[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Compute as einsum
    T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_hat, T, atol=tol)


def _try_rank(n, m, p, rank, T, deadline, max_iter=5000, seed=42):
    """Try to find a rank-`rank` decomposition via random init + gradient descent."""
    rng = np.random.RandomState(seed)
    dim_a, dim_b, dim_c = n * m, m * p, p * n
    
    # Random initialization
    U = rng.randn(dim_a, rank) * 0.5
    V = rng.randn(dim_b, rank) * 0.5
    W = rng.randn(dim_c, rank) * 0.5
    
    # Normalize initial columns to avoid scaling issues
    for r in range(rank):
        norm = np.sqrt(np.sum(U[:, r]**2) + np.sum(V[:, r]**2) + np.sum(W[:, r]**2))
        if norm > 1e-10:
            U[:, r] /= norm
            V[:, r] /= norm
            W[:, r] /= norm
    
    # Simple gradient descent / alternating minimization
    # We'll use a simple approach: alternate between fixing two matrices and solving for the third
    
    lr = 0.01
    best_err = np.inf
    
    for it in range(max_iter):
        if time.time() > deadline - 2:
            break
        
        # Compute current error
        T_hat = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.sum((T_hat - T)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-10:
            break
        
        # Gradient computation
        # dT_hat/dr = U[:,r] * V[:,r] * W[:,r]
        # grad_U[i,r] = sum_{j,k} (T_hat[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
        # grad_V[j,r] = sum_{i,k} (T_hat[i,j,k] - T[i,j,k]) * U[i,r] * W[k,r]
        # grad_W[k,r] = sum_{i,j} (T_hat[i,j,k] - T[i,j,k]) * U[i,r] * V[j,r]
        
        diff = T_hat - T
        
        grad_U = np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if err > best_err * 1.5:
            lr *= 0.5
        elif err < best_err * 0.8:
            lr = min(lr * 1.1, 0.1)
    
    # Check best result
    T_hat_best = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
    if np.allclose(T_hat_best, T, atol=1e-6):
        return best_U, best_V, best_W, True
    
    return None, None, None, False


def construct():
    n, m, p = 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    T = _build_tensor(n, m, p)
    
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    best_rank = n * m * p  # Fallback to schoolbook rank 48
    
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for each rank
        for seed in range(0, 20):
            if time.time() > deadline - 5:
                break
            
            U, V, W, success = _try_rank(n, m, p, rank, T, deadline, max_iter=3000, seed=seed)
            
            if success:
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback to schoolbook decomposition
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
