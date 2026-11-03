import numpy as np
import time

n, m, p = 3, 4, 8
nm, mp, pn = n * m, m * p, p * n

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matmul tensor with half-integer entries."""
    # Check entries are half-integers (within floating point tolerance)
    for M in (U, V, W):
        M2 = np.array(M, dtype=np.float64) * 2.0
        if not np.allclose(M2, np.round(M2), atol=1e-7):
            return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integer(M):
    """Snap matrix entries to nearest 0.5."""
    return np.round(np.array(M, dtype=np.float64) * 2.0) / 2.0

def _construct_schoolbook(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _construct_kron_decomposition(n, m, p):
    """
    Construct a decomposition using Kronecker product structure.
    Split dimensions into smaller blocks and use tensor products.
    """
    # For 3x4x8: split as 3=1*3, 4=1*4, 8=1*8
    # Or use 2x2 blocks where possible
    # For 3x4x8, we can use:
    # - 1x1x8 blocks (8 terms)
    # - 1x4x8 block (32 terms) -- too large
    # - 2x2x2 blocks (4 terms each)
    
    # Better approach: use the fact that 4 = 2*2 and 8 = 2*4
    # Split as (1,2) x (2,2) x (4,2)
    # This gives a rank of at most:
    # For each pair of splits, use Strassen-like or schoolbook
    
    # Simplest: use schoolbook for the 2x2x2 subproblem and combine
    # Actually, let's just use the schoolbook as fallback
    
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _residual_tensor(U, V, W, n, m, p):
    """Compute the residual tensor: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1"""
    R = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
    return T

def _try_rank_with_gradient_descent(R, n, m, p, time_limit, seed_base=0):
    """
    Try to find a rank-R decomposition using gradient descent.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        seed = seed_base + attempt * 137
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # Gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Compute residuals
            T = _residual_tensor(U, V, W, n, m, p)
            max_res = np.max(np.abs(T))
            
            if max_res < 1e-12:
                break
            
            # Gradient for U
            grad_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_U[i*m+j] += -T[i,j,k] * V[j*p+k] * W[k*n+i]
            
            # Gradient for V
            grad_V = np.zeros_like(V)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_V[j*p+k] += -T[i,j,k] * U[i*m+j] * W[k*n+i]
            
            # Gradient for W
            grad_W = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        grad_W[k*n+i] += -T[i,j,k] * U[i*m+j] * V[j*p+k]
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 0:
                effective_lr = lr * max_res / max(grad_norm, 1e-10)
                effective_lr = min(effective_lr, 0.1)
            else:
                effective_lr = 0.0
            
            U += effective_lr * grad_U
            V += effective_lr * grad_V
            W += effective_lr * grad_W
        
        # Snap to half-integers
        U_snap = _snap_to_half_integer(U)
        V_snap = _snap_to_half_integer(V)
        W_snap = _snap_to_half_integer(W)
        
        if _verify(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s limit
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = n * m * p
    
    # Search for lower rank using gradient descent
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(12, 32, 24) = 32
    min_rank = max(n*m, m*p, n*p)
    
    for R in range(min_rank, best_rank + 1):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 1:
            break
            
        # Give more time to lower ranks (harder problems)
        time_per_rank = min(10.0, time_left / max(1, best_rank - R + 1))
        
        result = _try_rank_with_gradient_descent(R, n, m, p, time_limit=time_per_rank, seed_base=R)
        
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_result = {
                "U": U_sol.tolist(),
                "V": V_sol.tolist(),
                "W": W_sol.tolist()
            }
            best_rank = R
            return best_result
    
    return best_result

DIMENSIONS = [3, 4, 8]
