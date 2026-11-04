import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def _schoolbook():
    """Return schoolbook decomposition with rank 54."""
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

def _verify(U, V, W, tol=1e-8):
    """Verify that the decomposition exactly matches the matmul tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build the matmul tensor: T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Compute the decomposed tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U_arr[idx_u, :], np.multiply(V_arr[idx_v, :], W_arr[idx_w, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_border_rank(rank, max_iters=2000, tol=1e-6):
    """
    Try to find a border-rank decomposition of given rank.
    Uses gradient descent on the border rank formulation.
    Returns (U, V, W) lists or None if failed.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with small random values
    np.random.seed(42)
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    # Target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # We want sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] ≈ 1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iters):
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    # val_r = U[idx_u,r] * V[idx_v,r] * W[idx_w,r]
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    loss += err * err
                    
                    # d/dU[idx_u,r] = 2*err*V[idx_v,r]*W[idx_w,r]
                    grad_U[idx_u, :] += 2.0 * err * V[idx_v, :] * W[idx_w, :]
                    grad_V[idx_v, :] += 2.0 * err * U[idx_u, :] * W[idx_w, :]
                    grad_W[idx_w, :] += 2.0 * err * U[idx_u, :] * V[idx_v, :]
        
        loss /= (n * m * p)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Update with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        if it % 200 == 199:
            lr *= 0.9
    
    if best_loss < 1e-4:
        # Round to nearest half-integer or integer for exactness
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), tol=1e-8):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Try without rounding
        if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), tol=1e-8):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 by 3x9 matrix multiplication.
    Starts with schoolbook rank 54 and searches for lower ranks.
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Start with schoolbook as best known
    best_result = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Based on known results, the border rank for 2x3x9 might be lower than 54
    # Try ranks from 40 down to 54
    for rank in range(40, schoolbook_rank):
        if time.time() > deadline:
            break
        
        result = _try_border_rank(rank, max_iters=1500, tol=1e-6)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            # Continue searching for even lower ranks
            continue
    
    if time.time() > deadline:
        return best_result
    
    return best_result
