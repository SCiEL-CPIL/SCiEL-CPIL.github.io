"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-18 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, tol=1e-9):
    """Check if the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Expected: T[i,j,k] = 1 if this is a valid mult tensor entry
    # Actually, the matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k)
    # Wait, no. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank_decomposition(rank, time_budget=10, max_iters=5000):
    """Try to find a rank-`rank` decomposition using random initialization and gradient descent."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Use Adam-like optimizer
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_U = np.zeros_like(U)
    v_V = np.zeros_like(V)
    v_W = np.zeros_like(W)
    t = 0
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        t += 1
        # Compute loss: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute the sum
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = val - 1.0
                    loss += err * err
                    
                    # Gradients
                    for r in range(rank):
                        grad_U[i * m + j, r] += 2 * err * V[j * p + k, r] * W[k * n + i, r]
                        grad_V[j * p + k, r] += 2 * err * U[i * m + j, r] * W[k * n + i, r]
                        grad_W[k * n + i, r] += 2 * err * U[i * m + j, r] * V[j * p + k, r]
        
        # Adam update
        for param, grad, m_buf, v_buf in [(U, grad_U, m_U, v_U),
                                           (V, grad_V, m_V, v_V),
                                           (W, grad_W, m_W, v_W)]:
            m_buf[:] = beta1 * m_buf + (1 - beta1) * grad
            v_buf[:] = beta2 * v_buf + (1 - beta2) * grad ** 2
            m_hat = m_buf / (1 - beta1 ** t)
            v_hat = v_buf / (1 - beta2 ** t)
            param[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
    
    # Verify the best solution
    if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist()):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None


def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # First, try to find a lower-rank decomposition
    # The schoolbook rank is 18. We know that for 2x3 * 3x3, the lower bound is at least
    # max(n*m, m*p, p*n) = max(6, 9, 6) = 9, but actually it's higher.
    # Let's try ranks from 12 upward (since rank < 18 would be an improvement)
    
    # Actually, let's first just return the schoolbook decomposition as a safe fallback
    schoolbook = _schoolbook()
    
    # Try to find a better decomposition with limited time
    for rank in range(12, 18):
        if time.time() > deadline - 20:
            break
        result = _try_rank_decomposition(rank, time_budget=15, max_iters=3000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If we didn't find a better one, return schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
