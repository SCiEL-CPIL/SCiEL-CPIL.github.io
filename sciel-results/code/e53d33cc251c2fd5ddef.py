"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
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

def _verify_decomp(U, V, W):
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
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _compute_loss(U, V, W):
    """Vectorized computation of the decomposition loss."""
    n, m, p = DIMENSIONS
    # Build index arrays for vectorized access
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                loss += (val - 1.0) ** 2
    return loss

def _compute_loss_grad(U, V, W):
    """Compute loss and gradients for U, V, W."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Precompute products
    UV = np.zeros((n*m, m*p, rank))
    for i in range(n*m):
        for j in range(m*p):
            UV[i, j, :] = U[i, :] * V[j, :]
    
    loss = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                err = val - 1.0
                loss += err * err
                # Gradient w.r.t. U[idx_u, r] = err * V[idx_v, r] * W[idx_w, r]
                dU[idx_u, :] += 2 * err * V[idx_v, :] * W[idx_w, :]
                # Gradient w.r.t. V[idx_v, r] = err * U[idx_u, r] * W[idx_w, r]
                dV[idx_v, :] += 2 * err * U[idx_u, :] * W[idx_w, :]
                # Gradient w.r.t. W[idx_w, r] = err * U[idx_u, r] * V[idx_v, r]
                dW[idx_w, :] += 2 * err * U[idx_u, :] * V[idx_v, :]
    
    return loss, dU, dV, dW

def _try_gradient_descent(target_rank, max_iter=5000, tol=1e-10, deadline=140.0, start_time=0):
    """Try to find a decomposition of given rank using gradient descent."""
    n, m, p = DIMENSIONS
    
    # Try multiple random initializations
    for seed in range(10):
        if time.time() > deadline:
            return None
        
        np.random.seed(seed + target_rank * 1000)
        
        # Initialize with small random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Adam optimizer
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = 0.01
        
        for step in range(max_iter):
            if time.time() > deadline:
                return None
            
            t += 1
            loss, dU, dV, dW = _compute_loss_grad(U, V, W)
            
            if loss < tol:
                if _verify_decomp(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return None
            
            # Adam update
            lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * dU**2
            U -= lr_t * mU / (np.sqrt(vU) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * dV**2
            V -= lr_t * mV / (np.sqrt(vV) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * dW**2
            W -= lr_t * mW / (np.sqrt(vW) + eps)
            
            # Decay learning rate
            if step % 1000 == 999:
                lr *= 0.5
        
        # Check final
        if loss < tol:
            if _verify_decomp(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _try_strassen_based():
    """
    Try to construct a decomposition using Strassen's algorithm ideas.
    
    For 3x5x5, we can try to embed into a larger problem and use
    fast matrix multiplication techniques.
    
    One approach: Pad to 4x8x8 and use known fast algorithms,
    then extract the 3x5x5 part.
    """
    n, m, p = DIMENSIONS
    
    # Try to use the fact that 3x5x5 can be related to smaller problems
    # by block decomposition.
    
    # A is 3x5, B is 5x5
    # Split A into A1 (3x3) and A2 (3x2)
    # Split B into B1 (3x5) and B2 (2x5)
    # Then C = A1*B1 + A2*B2
    
    # A1*B1 is 3x3 * 3x5 = 3x5 (rank 27 schoolbook, can be done in less)
    # A2*B2 is 3x2 * 2x5 = 3x5 (rank 30 schoolbook)
    
    # This doesn't immediately help. Let's try another approach.
    
    # Use the identity that matmul(n,m,p) has rank at most n*m*p - something
    # for small cases.
    
    # For now, return None to use gradient descent
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Try to find a decomposition with rank below schoolbook
    # Use gradient descent for various target ranks
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 25
    # Try ranks from 50 down to 40
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try gradient descent for various ranks
    for target_rank in range(60, 39, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_gradient_descent(target_rank, max_iter=3000, tol=1e-10, deadline=deadline - 5, start_time=start_time)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Found a valid decomposition, try to go lower
            continue
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
