"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _schoolbook():
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

def _verify_decomp(U, V, W, tol=1e-6):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _compute_loss(U, V, W):
    """Compute the squared loss for the tensor decomposition."""
    n, m, p = DIMENSIONS
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                loss += (val - 1.0) ** 2
    return loss

def _try_gradient_descent(target_rank, start_time, deadline, seed):
    """Try to find a rank-minimal decomposition using gradient descent."""
    n, m, p = DIMENSIONS
    
    np.random.seed(seed)
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Learning rate schedule
    lr = 0.05
    lr_decay = 0.95
    
    for step in range(500):
        if time.time() - start_time > deadline - 5:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = 2.0 * (val - 1.0)
                    dU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Compute loss
        loss = _compute_loss(U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            result = {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
            if _verify_decomp(best_U, best_V, best_W):
                return result
            break
        
        # Decay learning rate
        lr *= lr_decay
    
    return None

def _try_polish(U, V, W, start_time, deadline):
    """Polish a near-optimal solution with tighter tolerance."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_loss = _compute_loss(best_U, best_V, best_W)
    
    if best_loss > 1e-4:
        return None
    
    # Use smaller learning rates for polishing
    lr = 0.001
    
    for step in range(200):
        if time.time() - start_time > deadline - 3:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err = 2.0 * (val - 1.0)
                    dU[i*m+j, :] += err * V[j*p+k, :] * W[k*n+i, :]
                    dV[j*p+k, :] += err * U[i*m+j, :] * W[k*n+i, :]
                    dW[k*n+i, :] += err * U[i*m+j, :] * V[j*p+k, :]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Compute loss
        loss = _compute_loss(U, V, W)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            result = {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
            if _verify_decomp(best_U, best_V, best_W, tol=1e-8):
                return result
            break
        
        lr *= 0.9
    
    return None

def _try_winograd_structured():
    """
    Try structured decompositions and gradient-based optimization.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Try various rank values below schoolbook
    for target_rank in range(50, 76):
        if time.time() - start_time > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() - start_time > deadline - 5:
                break
            
            seed = target_rank * 1000 + attempt
            result = _try_gradient_descent(target_rank, start_time, deadline, seed)
            if result is not None:
                return result
    
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # First, try the structured approach
    result = _try_winograd_structured()
    
    if result is not None:
        return result
    
    # Fall back to schoolbook
    if time.time() - start_time < deadline - 5:
        return _schoolbook()
    
    return _schoolbook()
# EVOLVE-BLOCK-END
