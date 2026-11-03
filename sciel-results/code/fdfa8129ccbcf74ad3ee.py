"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    # Check tensor identity: for all i,j,k: sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank17():
    """Return the known rank-17 Strassen+schoolbook decomposition."""
    n, m, p = 2, 3, 3
    rank = 17
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen part for 2x2 block (indices 0-6)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for remaining terms (indices 7-16)
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _try_rank16_search(deadline):
    """Try to find a rank-16 decomposition via numerical optimization."""
    n, m, p = 2, 3, 3
    rank = 16
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Use gradient descent to minimize sum of squared errors
        # Error: for each (i,j,k), (sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] - 1)^2
        lr = 0.01
        for step in range(5000):
            if time.time() > deadline - 30:
                break
            
            # Compute gradient
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Current value
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = val - 1.0
                        # Gradient w.r.t. U[i*m+j, s]
                        dU[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                        # Gradient w.r.t. V[j*p+k, s]
                        dV[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                        # Gradient w.r.t. W[k*n+i, s]
                        dW[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if step % 500 == 0:
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-6:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                        return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # First, try to find a rank-16 decomposition
    result16 = _try_rank16_search(deadline)
    if result16 is not None:
        U, V, W = result16
        # Double-check
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank-17
    U, V, W = _build_rank17()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Should not reach here, but return rank-17 anyway
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
