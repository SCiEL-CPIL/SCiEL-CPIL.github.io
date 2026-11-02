"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly computes matrix multiplication.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _gradient_descent_search(n, m, p, target_rank, deadline, max_iter=5000, lr=0.01):
    """
    Use gradient descent to find a rank-`target_rank` decomposition.
    Objective: minimize sum over (i,j,k) of (sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] - 1)^2
    """
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m+j]
                    v_row = V[j*p+k]
                    w_row = W[k*n+i]
                    
                    val = np.dot(u_row, v_row * w_row)
                    err = val - 1.0
                    loss += err * err
                    
                    # Gradients
                    grad_U[i*m+j] += 2 * err * (v_row * w_row)
                    grad_V[j*p+k] += 2 * err * (u_row * w_row)
                    grad_W[k*n+i] += 2 * err * (u_row * v_row)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
    
    if _verify(best_U, best_V, best_W, n, m, p, tol=1e-8):
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # 1. Schoolbook Decomposition (Rank 100)
    U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # 2. Attempt to find lower rank using gradient-based search
    # Try ranks from 99 down to 80, with increasing time allocation
    for target_rank in range(99, 79, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            result = _gradient_descent_search(n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        
        if found:
            break
    
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
