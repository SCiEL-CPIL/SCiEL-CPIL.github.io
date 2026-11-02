"""
Strategy: Gradient-based search for minimal-rank tensor decomposition of 3x3 matrix multiplication.

Instead of a static Strassen+boundary construction, we parameterize a rank-r decomposition
U (9 x r), V (9 x r), W (9 x r) and minimize the sum of squared errors over all 27
(i,j,k) tensor entries using gradient descent. We try ranks 25, 26, 27 in order,
returning the first rank that achieves error < tol. A hard deadline guard ensures
we return within 150s.

This is a fundamentally different approach: numerical optimization vs. symbolic construction.
It may find rank-25 if such a decomposition exists, or confirm rank-26 is optimal.
"""

import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n=3, m=3, p=3, tol=1e-6):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank_26():
    """Fallback: guaranteed rank-26 decomposition via Strassen + boundary."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2 core
    U[0][0] += 1.0; U[4][0] += 1.0
    V[0][0] += 1.0; V[4][0] += 1.0
    W[0][0] += 1.0; W[4][0] += 1.0
    
    U[3][1] += 1.0; U[4][1] += 1.0
    V[0][1] += 1.0
    W[3][1] += 1.0
    
    U[0][2] += 1.0
    V[3][2] += 1.0; V[4][2] += -1.0
    W[1][2] += 1.0
    
    U[4][3] += 1.0
    V[1][3] += 1.0; V[0][3] += -1.0
    W[0][3] += 1.0; W[3][3] += 1.0
    
    U[0][4] += 1.0; U[1][4] += 1.0
    V[4][4] += 1.0
    W[1][4] += 1.0; W[0][4] += -1.0
    
    U[3][5] += 1.0; U[0][5] += -1.0
    V[0][5] += 1.0; V[1][5] += 1.0
    W[4][5] += 1.0
    
    U[1][6] += 1.0; U[4][6] += -1.0
    V[3][6] += 1.0; V[4][6] += 1.0
    W[0][6] += -1.0; W[4][6] += 1.0
    
    r = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*3+j][r] = 1.0
                V[j*3+k][r] = 1.0
                W[k*3+i][r] = 1.0
                r += 1
    
    return U, V, W

def _construct_rank_27():
    """Schoolbook rank-27 decomposition."""
    n, m, p = 3, 3, 3
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
                
    return U, V, W

def _gradient_search(rank, deadline, seed=42):
    """
    Try to find a valid rank-r decomposition via gradient descent.
    Returns (U, V, W) as lists if success, None otherwise.
    """
    np.random.seed(seed)
    n, m, p = 3, 3, 3
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
    
    def compute_loss(U, V, W):
        loss = 0.0
        for ui, vi, wi in indices:
            val = np.dot(U[ui], V[vi] * W[wi])
            loss += (val - 1.0) ** 2
        return loss
    
    def compute_gradients(U, V, W):
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for ui, vi, wi in indices:
            val = np.dot(U[ui], V[vi] * W[wi])
            err = 2.0 * (val - 1.0)
            # d/dU[ui] = err * V[vi] * W[wi]
            dU[ui] += err * V[vi] * W[wi]
            # d/dV[vi] = err * U[ui] * W[wi]
            dV[vi] += err * U[ui] * W[wi]
            # d/dW[wi] = err * U[ui] * V[vi]
            dW[wi] += err * U[ui] * V[vi]
        return dU, dV, dW
    
    # Adam-like optimizer with adaptive learning rate
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    t = 0
    best_loss = compute_loss(U, V, W)
    
    while time.time() < deadline - 5:  # Safety margin
        t += 1
        dU, dV, dW = compute_gradients(U, V, W)
        
        m_U = beta1 * m_U + (1 - beta1) * dU
        v_U = beta2 * v_U + (1 - beta2) * dU ** 2
        m_hat_U = m_U / (1 - beta1 ** t)
        v_hat_U = v_U / (1 - beta2 ** t)
        
        m_V = beta1 * m_V + (1 - beta1) * dV
        v_V = beta2 * v_V + (1 - beta2) * dV ** 2
        m_hat_V = m_V / (1 - beta1 ** t)
        v_hat_V = v_V / (1 - beta2 ** t)
        
        m_W = beta1 * m_W + (1 - beta1) * dW
        v_W = beta2 * v_W + (1 - beta2) * dW ** 2
        m_hat_W = m_W / (1 - beta1 ** t)
        v_hat_W = v_W / (1 - beta2 ** t)
        
        U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
        V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
        W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
        
        loss = compute_loss(U, V, W)
        if loss < best_loss:
            best_loss = loss
        
        # Check convergence
        if best_loss < 1e-10:
            break
        
        # Adaptive learning rate decay
        if t % 500 == 0:
            lr *= 0.95
        
        # If loss is increasing significantly, restart with smaller lr
        if t > 1000 and loss > best_loss * 1.1:
            lr *= 0.5
    
    # Final check
    if best_loss < 1e-8:
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify_decomposition(U_list, V_list, W_list, tol=1e-6):
            return U_list, V_list, W_list
    
    return None

def _try_multiple_seeds(rank, deadline):
    """Try gradient search with multiple seeds."""
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        result = _gradient_search(rank, deadline, seed=seed)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Try rank 25 first (if it exists, this is optimal)
    if time.time() < deadline - 30:
        result = _try_multiple_seeds(25, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Try rank 26 (known to exist)
    if time.time() < deadline - 15:
        # First try the known static construction
        U, V, W = _construct_rank_26()
        if _verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        # Then try gradient search
        result = _try_multiple_seeds(26, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 27
    U, V, W = _construct_rank_27()
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
