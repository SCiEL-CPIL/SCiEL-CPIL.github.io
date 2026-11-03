"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from numpy import linalg as la
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Check against the multiplication tensor
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                # The true value: A[i,j]*B[j,k] contributes to C[i,k]
                # In the tensor, T[i,j,k] = 1 for all valid (i,j,k)
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_rank_r(n, m, p, r, max_iters=500, seed=42):
    """Try to find a rank-r decomposition using iterative refinement."""
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n*m, r)
    V = rng.randn(m*p, r)
    W = rng.randn(p*n, r)
    
    # Normalize columns to prevent scaling issues
    for col in range(r):
        norms = [np.linalg.norm(U[:, col]), np.linalg.norm(V[:, col]), np.linalg.norm(W[:, col])]
        if norms[0] > 0 and norms[1] > 0 and norms[2] > 0:
            scale = (norms[0] * norms[1] * norms[2]) ** (1.0/3.0)
            U[:, col] /= scale
            V[:, col] /= scale
            W[:, col] /= scale
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Alternating least squares
        # Update U: for each (i,j), minimize sum_k (T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # This is a linear system in U[i*m+j,:]
        
        # Precompute for efficiency
        # For fixed (i,j), the error is:
        # sum_k [1 - sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])]^2
        # Let M_{k,r} = V[j*p+k,r] * W[k*n+i,r]
        # Then we want to minimize sum_k (1 - U[i*m+j,:] @ M[k,:])^2
        # This is a least squares problem: minimize ||M^T @ U[i*m+j,:] - 1||^2
        # Solution: U[i*m+j,:] = (M M^T)^{-1} M @ 1
        
        # Similarly for V and W
        
        # Due to complexity, use a simpler gradient-based approach
        # Compute gradient of error with respect to U, V, W
        
        # Error E = sum_{i,j,k} (1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])^2
        # dE/dU[i*m+j,r] = -2 * sum_k (1 - ...) * V[j*p+k,r] * W[k*n+i,r]
        
        lr = 0.01
        E = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute inner product
                    inner = 0.0
                    for r in range(r):
                        inner += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    diff = 1.0 - inner
                    E += diff * diff
                    for r in range(r):
                        dU[i*m+j, r] += -2.0 * diff * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += -2.0 * diff * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += -2.0 * diff * U[i*m+j, r] * V[j*p+k, r]
        
        # Gradient descent step
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Normalize
        for col in range(r):
            norms = [np.linalg.norm(U[:, col]), np.linalg.norm(V[:, col]), np.linalg.norm(W[:, col])]
            if norms[0] > 1e-10 and norms[1] > 1e-10 and norms[2] > 1e-10:
                scale = (norms[0] * norms[1] * norms[2]) ** (1.0/3.0)
                U[:, col] /= scale
                V[:, col] /= scale
                W[:, col] /= scale
        
        if E < best_err:
            best_err = E
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_err
    return None, None, None, best_err

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Tries to find a rank lower than the schoolbook rank of 36 using numerical search.
    Falls back to schoolbook if no lower rank is found.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # First, try to find a lower rank
    for r in range(schoolbook_rank - 1, max(10, schoolbook_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            U, V, W, err = _try_rank_r(n, m, p, r, max_iters=200, seed=seed)
            if U is not None:
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
