"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _to_half_int(x, tol=1e-9):
    """Convert a float to the nearest half-integer (multiple of 0.5) if close enough."""
    # x should be close to k/2 for some integer k
    # Multiply by 2, round to nearest int
    y = x * 2.0
    k = int(round(y))
    if abs(y - k) < tol:
        return k / 2.0
    return None

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    """
    rank = U.shape[1]
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each (i, j, k), verify: sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U[i*m+j, :], V[j*p+k, :])
                rhs = W[k*n+i, :]
                if not np.allclose(lhs, rhs, atol=tol, rtol=tol):
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_rank(n, m, p, rank, deadline, max_iters=5000, lr=0.01):
    """Try to find a rank decomposition using numerical optimization.
    Returns (U, V, W) as lists of lists with half-integer entries, or None.
    """
    if time.time() > deadline:
        return None
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Compute loss: for each (i,j,k), ||U[i*m+j,:] * V[j*p+k,:] - W[k*n+i,:]||^2
    def compute_loss(U, V, W):
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = np.dot(U[i*m+j, :], V[j*p+k, :]) - W[k*n+i, :]
                    loss += np.dot(diff, diff)
        return loss
    
    initial_loss = compute_loss(U, V, W)
    
    # Simple gradient descent
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = np.dot(U[i*m+j, :], V[j*p+k, :]) - W[k*n+i, :]
                    dU[i*m+j, :] += 2 * np.dot(diff, V[j*p+k, :])
                    dV[j*p+k, :] += 2 * np.dot(diff, U[i*m+j, :])
                    dW[k*n+i, :] -= 2 * diff
        
        # Gradient step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if it % 100 == 0:
            loss = compute_loss(U, V, W)
            if loss < 1e-12:
                break
    
    # Try to round to half-integers
    U_h = np.zeros_like(U)
    V_h = np.zeros_like(V)
    W_h = np.zeros_like(W)
    
    success = True
    for idx in range(U.size):
        val = _to_half_int(U.flat[idx])
        if val is None:
            success = False
            break
        U_h.flat[idx] = val
    if not success:
        return None
    
    for idx in range(V.size):
        val = _to_half_int(V.flat[idx])
        if val is None:
            success = False
            break
        V_h.flat[idx] = val
    if not success:
        return None
    
    for idx in range(W.size):
        val = _to_half_int(W.flat[idx])
        if val is None:
            success = False
            break
        W_h.flat[idx] = val
    if not success:
        return None
    
    # Verify the exact decomposition
    if _verify(U_h, V_h, W_h, n, m, p, tol=1e-9):
        # Convert to lists
        U_list = U_h.tolist()
        V_list = V_h.tolist()
        W_list = W_h.tolist()
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook (rank 42)
    U, V, W = _build_schoolbook(n, m, p)
    best_result = (U.tolist(), V.tolist(), W.tolist())
    best_rank = n * m * p
    
    # Try lower ranks
    # The theoretical minimum rank for 2x3 x 3x7 is at least max(2*3, 3*7, 2*7) = 21
    # Known results suggest rank around 30-35 might be achievable
    for rank in range(21, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, deadline, max_iters=2000, lr=0.05)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = (U_list, V_list, W_list)
            best_rank = rank
            # If we found a valid decomposition, try even lower
            continue
    
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
# EVOLVE-BLOCK-END
