"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 4, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)

def _compute_error(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.max(np.abs(T - 1.0))

def _try_lp_rank(rank, seed=42, deadline=None):
    """Try to find a rank-r decomposition using linear programming with rational constraints."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # Variables: U (n*m*rank), V (m*p*rank), W (p*n*rank)
    # Total variables: (n*m + m*p + p*n) * rank
    num_vars = (n * m + m * p + p * n) * rank
    
    # We want to minimize ||T - 1|| where T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a non-convex problem, so we use a different approach:
    # Fix one matrix and solve for others, or use structured initialization
    
    # Better approach: Use a grid search over small integer values
    # Try values in {-1, 0, 1} for simplicity, then expand if needed
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    # Try structured patterns with small integer values
    for attempt in range(50):
        if deadline and time.time() > deadline:
            break
        
        # Initialize with small random integers
        U_vals = rng.randint(-2, 3, size=(n * m, rank))
        V_vals = rng.randint(-2, 3, size=(m * p, rank))
        W_vals = rng.randint(-2, 3, size=(p * n, rank))
        
        # Scale to get better conditioning
        scale = 0.5
        U = U_vals.astype(float) * scale
        V = V_vals.astype(float) * scale
        W = W_vals.astype(float) * scale
        
        err = _compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    
    return None

def _try_rank(rank, seed=42, max_iters=100, deadline=None):
    """Try to find a rank-rank decomposition using hybrid LP + local search."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    # First try LP-based approach
    result = _try_lp_rank(rank, seed=seed, deadline=deadline)
    if result is not None:
        return result
    
    # Fallback: local search with structured initialization
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Structured random initialization
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        # Local optimization using gradient descent
        for step in range(30):
            if deadline and time.time() > deadline:
                break
            
            # Compute gradients
            Ur = U.reshape(n, m, rank)
            Vr = V.reshape(m, p, rank)
            Wr = W.reshape(p, n, rank)
            
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            err = np.max(np.abs(T - 1.0))
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Simple gradient step
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    grad_U = np.zeros(rank)
                    for k in range(p):
                        grad_U += Vr[j, k, :] * Wr[k, i, :] * (T[i, j, k] - 1.0)
                    U[i*m+j, :] -= lr * grad_U
            
            for j in range(m):
                for k in range(p):
                    grad_V = np.zeros(rank)
                    for i in range(n):
                        grad_V += Ur[i, j, :] * Wr[k, i, :] * (T[i, j, k] - 1.0)
                    V[j*p+k, :] -= lr * grad_V
            
            for k in range(p):
                for i in range(n):
                    grad_W = np.zeros(rank)
                    for j in range(m):
                        grad_W += Ur[i, j, :] * Vr[j, k, :] * (T[i, j, k] - 1.0)
                    W[k*n+i, :] -= lr * grad_W
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-6:
        # Snap to half-integers
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        err_after = _compute_error(best_U, best_V, best_W)
        if err_after < 1e-6:
            return best_U, best_V, best_W
    
    return None

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    
    # Try ranks from lower bound upward
    for rank in range(50, 72):
        if time.time() > deadline - 5:
            break
        
        # Try multiple seeds for this rank
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            result = _try_rank(rank, seed=seed + rank * 100, max_iters=30, deadline=deadline - 5)
            if result is not None:
                U, V, W = result
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
