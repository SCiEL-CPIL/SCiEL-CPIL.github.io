"""Rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication.
Uses a structured search starting from schoolbook rank 45 and attempting
to find lower-rank decompositions via random initialization and local search."""

import time
import numpy as np

DIMENSIONS = [3, 3, 5]


def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(actual - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
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


def _random_init(n, m, p, rank, seed=None):
    """Random initialization of factor matrices."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Normalize to avoid numerical issues
    for c in range(rank):
        norm = np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c])
        if norm > 1e-10:
            scale = (1.0 / norm) ** (1.0 / 3.0)
            U[:, c] *= scale
            V[:, c] *= scale
            W[:, c] *= scale
    
    return U, V, W


def _compute_residual(U, V, W, n, m, p):
    """Compute the residual tensor T - sum_c U_c ⊗ V_c ⊗ W_c."""
    rank = U.shape[1]
    residual = np.ones((n, m, p))
    
    for c in range(rank):
        residual -= np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    
    return residual


def _local_search(U, V, W, n, m, p, max_iters=100, tol=1e-10, time_limit=10):
    """Perform local search to reduce the residual norm."""
    start_time = time.time()
    rank = U.shape[1]
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute residual
        residual = _compute_residual(U, V, W, n, m, p)
        residual_norm = np.linalg.norm(residual)
        
        if residual_norm < tol:
            break
        
        # For each column, try to optimize it
        for c in range(rank):
            # Hold other columns fixed, optimize column c
            # The contribution of column c is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
            # We want to minimize ||residual - U[:,c] ⊗ V[:,c] ⊗ W[:,c]||
            
            # This is a non-convex problem, so use a simple gradient step
            # Compute gradients with respect to U[:,c], V[:,c], W[:,c]
            
            # Gradient wrt U[i, c]: sum_{j,k} residual[i,j,k] * V[j,c] * W[k,c]
            grad_U = np.zeros(n * m)
            grad_V = np.zeros(m * p)
            grad_W = np.zeros(p * n)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = residual[i, j, k]
                        grad_U[i * m + j] += val * V[j * p + k, c] * W[k * n + i, c]
                        grad_V[j * p + k] += val * U[i * m + j, c] * W[k * n + i, c]
                        grad_W[k * n + i] += val * U[i * m + j, c] * V[j * p + k, c]
            
            # Simple gradient descent step
            lr = 0.01
            U[:, c] -= lr * grad_U
            V[:, c] -= lr * grad_V
            W[:, c] -= lr * grad_W
            
            # Renormalize
            norm = np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c])
            if norm > 1e-10:
                scale = (1.0 / norm) ** (1.0 / 3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
    
    return U, V, W, residual_norm


def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=20):
    """Try to find a decomposition of rank target_rank."""
    start_time = time.time()
    
    # Try multiple random initializations
    best_residual = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
        
        U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=42 + attempt)
        
        U_opt, V_opt, W_opt, res_norm = _local_search(
            U_init.copy(), V_init.copy(), W_init.copy(), 
            n, m, p, max_iters=50, time_limit=time_limit / 5
        )
        
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U_opt, V_opt, W_opt
    
    if best_residual < 1e-8 and best_U is not None:
        return best_U, best_V, best_W
    return None


def construct():
    deadline = time.time() + 140  # 140s budget, 10s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    if not _verify_numpy(U, V, W, n, m, p):
        raise RuntimeError("Schoolbook verification failed")
    
    best_rank = n * m * p  # 45
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower-rank decompositions
    # Known lower bounds suggest rank around 30-40 might be possible
    # Try ranks from 44 down to 30
    
    for target_rank in range(44, 29, -1):
        if time.time() > deadline - 30:  # Stop if only 30s left
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=15)
        
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_numpy(U_new, V_new, W_new, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                break  # Found a valid lower-rank decomposition
    
    # If no improvement found, try a more aggressive search with more iterations
    if best_rank == 45 and time.time() < deadline - 20:
        for target_rank in range(44, 35, -1):
            if time.time() > deadline - 10:
                break
            
            # Try with more random initializations
            start_time = time.time()
            best_residual = np.inf
            best_U2, best_V2, best_W2 = None, None, None
            
            for attempt in range(10):
                if time.time() - start_time > 10:
                    break
                
                U_init, V_init, W_init = _random_init(n, m, p, target_rank, seed=100 + attempt)
                U_opt, V_opt, W_opt, res_norm = _local_search(
                    U_init.copy(), V_init.copy(), W_init.copy(),
                    n, m, p, max_iters=100, time_limit=1
                )
                
                if res_norm < best_residual:
                    best_residual = res_norm
                    best_U2, best_V2, best_W2 = U_opt, V_opt, W_opt
            
            if best_residual < 1e-8 and best_U2 is not None:
                if _verify_numpy(best_U2, best_V2, best_W2, n, m, p):
                    best_rank = target_rank
                    best_U, best_V, best_W = best_U2, best_V2, best_W2
                    break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
