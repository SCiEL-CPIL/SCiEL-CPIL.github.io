"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_rank_r(n, m, p, r, max_time=10):
    """Try to find a rank-r decomposition."""
    start_time = time.time()
    
    # Initialize with random values
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    # Use gradient descent on the constraint:
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Minimize: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    lr = 0.01
    best_loss = float('inf')
    
    for iteration in range(2000):
        if time.time() - start_time > max_time:
            break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U[i*m+j, :]
                    vj = V[j*p+k, :]
                    wk = W[k*n+i, :]
                    val = np.dot(ui, vj * wk)
                    loss += (val - 1.0) ** 2
                    err = val - 1.0
                    
                    # Gradient w.r.t. U[i*m+j, c]
                    grad_U[i*m+j, :] += 2 * err * vj * wk
                    # Gradient w.r.t. V[j*p+k, c]
                    grad_V[j*p+k, :] += 2 * err * ui * wk
                    # Gradient w.r.t. W[k*n+i, c]
                    grad_W[k*n+i, :] += 2 * err * ui * vj
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
        
        if best_loss < 1e-8:
            break
    
    if best_loss < 1e-6:
        # Round to half-integers and verify
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Based on known results, try ranks from lower bound up
    # Lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually the tensor rank lower bound is more complex
    
    # Try structured approaches first
    # For matrix multiplication, there are known algorithms with rank < n*m*p
    
    # Try rank from 50 down to 30 (practical range)
    for r in range(60, 29, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_r(n, m, p, r, max_time=10)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U, "V": V, "W": W}
            break  # Found a valid decomposition, return it
    
    return best_result
# EVOLVE-BLOCK-END
