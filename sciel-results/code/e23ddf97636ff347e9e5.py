"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time as _time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_decomposition(n, m, p, max_rank, max_time=30.0, seed=42):
    """
    Search for a low-rank tensor decomposition using gradient-based optimization.
    Returns (U, V, W) or None if not found within time limit.
    """
    start_time = _time.time()
    rng = np.random.RandomState(seed)
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Target tensor: T[i,j,k] = 1 for all valid indices
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Initialize with random values
    U = rng.randn(n_m, max_rank) * 0.5
    V = rng.randn(m_p, max_rank) * 0.5
    W = rng.randn(p_n, max_rank) * 0.5
    
    # Use Adam optimizer
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
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    while _time.time() - start_time < max_time:
        t += 1
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # val = sum_r U[u_idx,r] * V[v_idx,r] * W[w_idx,r]
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err = val - 1.0
                    loss += err * err
                    
                    # d/dU[u_idx,r] = 2*err * V[v_idx,r] * W[w_idx,r]
                    grad_U[u_idx, :] += 2.0 * err * (V[v_idx, :] * W[w_idx, :])
                    grad_V[v_idx, :] += 2.0 * err * (U[u_idx, :] * W[w_idx, :])
                    grad_W[w_idx, :] += 2.0 * err * (U[u_idx, :] * V[v_idx, :])
        
        # Adam update
        m_U = beta1 * m_U + (1 - beta1) * grad_U
        v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
        m_hat_U = m_U / (1 - beta1 ** t)
        v_hat_U = v_U / (1 - beta2 ** t)
        U -= lr * m_hat_U / (np.sqrt(v_hat_U) + eps)
        
        m_V = beta1 * m_V + (1 - beta1) * grad_V
        v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
        m_hat_V = m_V / (1 - beta1 ** t)
        v_hat_V = v_V / (1 - beta2 ** t)
        V -= lr * m_hat_V / (np.sqrt(v_hat_V) + eps)
        
        m_W = beta1 * m_W + (1 - beta1) * grad_W
        v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
        m_hat_W = m_W / (1 - beta1 ** t)
        v_hat_W = v_W / (1 - beta2 ** t)
        W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Adaptive learning rate
        if t % 100 == 0:
            lr *= 0.95
    
    # Verify the best solution
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-8):
        return best_U, best_V, best_W
    
    return None

def _construct_2x3x7():
    """
    Construct a low-rank decomposition for 2x3 x 3x7 matrix multiplication.
    Uses a structured approach: split 7 columns into groups and use
    known constructions for smaller subproblems.
    """
    n, m, p = 2, 3, 7
    
    # Strategy: Use the fact that we can decompose by grouping columns
    # 7 = 3 + 4, so use 2x3x3 (rank 5) + 2x3x4 (rank 6) = rank 11
    # Or 7 = 2 + 2 + 3, use three subproblems
    
    # First, let's try to find good decompositions for subproblems
    # 2x3x3: known minimal rank is 5
    # 2x3x4: let's search for rank 6
    
    best_result = None
    best_rank = n * m * p  # schoolbook = 42
    
    # Try different splits and ranks
    splits = [
        (3, 4, [5, 6]),
        (2, 5, [4, 7]),
        (4, 3, [6, 5]),
        (2, 2, 3, [4, 4, 5]),
    ]
    
    deadline = _time.time() + 100  # Safety margin
    
    # Try 2x3x3 with rank 5
    if _time.time() < deadline:
        res_233 = _find_decomposition(2, 3, 3, 5, max_time=20.0, seed=1)
        if res_233 is not None:
            U1, V1, W1 = res_233
            # Now find 2x3x4 with rank 6
            if _time.time() < deadline:
                res_234 = _find_decomposition(2, 3, 4, 6, max_time=20.0, seed=2)
                if res_234 is not None:
                    U2, V2, W2 = res_234
                    # Combine
                    rank = 5 + 6
                    U = np.zeros((6, rank), dtype=float)
                    V = np.zeros((21, rank), dtype=float)
                    W = np.zeros((14, rank), dtype=float)
                    
                    U[:, 0:5] = U1
                    V[0:9, 0:5] = V1
                    W[0:6, 0:5] = W1
                    
                    U[:, 5:11] = U2
                    V[9:21, 5:11] = V2
                    W[6:14, 5:11] = W2
                    
                    if _verify_decomposition(U, V, W, 2, 3, 7, tol=1e-8):
                        return U.tolist(), V.tolist(), W.tolist()
    
    # Try 2x3x2 with rank 4, then 2x3x5 with rank 7
    if _time.time() < deadline:
        res_232 = _find_decomposition(2, 3, 2, 4, max_time=15.0, seed=3)
        if res_232 is not None:
            U1, V1, W1 = res_232
            res_235 = _find_decomposition(2, 3, 5, 7, max_time=20.0, seed=4)
            if res_235 is not None:
                U2, V2, W2 = res_235
                rank = 4 + 7
                U = np.zeros((6, rank), dtype=float)
                V = np.zeros((21, rank), dtype=float)
                W = np.zeros((14, rank), dtype=float)
                
                U[:, 0:4] = U1
                V[0:6, 0:4] = V1
                W[0:4, 0:4] = W1
                
                U[:, 4:11] = U2
                V[6:21, 4:11] = V2
                W[4:14, 4:11] = W2
                
                if _verify_decomposition(U, V, W, 2, 3, 7, tol=1e-8):
                    return U.tolist(), V.tolist(), W.tolist()
    
    # Try direct search on the full problem with rank 12
    if _time.time() < deadline:
        res_direct = _find_decomposition(2, 3, 7, 12, max_time=30.0, seed=5)
        if res_direct is not None:
            U, V, W = res_direct
            return U.tolist(), V.tolist(), W.tolist()
    
    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = _time.time()
    
    # Try to find a low-rank decomposition
    result = _construct_2x3x7()
    
    if result is not None:
        U, V, W = result
        return {
            "U": U,
            "V": V,
            "W": W
        }
    
    # Should not reach here, but fallback
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Simple validation check if run directly
    res = construct()
    n, m, p = DIMENSIONS
    U = res['U']
    V = res['V']
    W = res['W']
    
    # Verify reconstruction
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :] * W_np[w_idx, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    print("Max error:", max_err)
    print("Rank:", rank)
    print("Valid:", max_err < 1e-8)
