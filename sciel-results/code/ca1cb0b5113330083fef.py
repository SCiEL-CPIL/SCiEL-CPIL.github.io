"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 3]

def _matmul_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] for (i in [0,n), j in [0,m), k in [0,p)).
    T is indexed by (i, j, k) where the contraction is sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k].
    For matrix multiplication, T[i,j,k] = 1 if this corresponds to an actual multiplication term.
    Actually, the tensor for matmul C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the tensor T[i,k,j] = 1 (all ones for valid indices).
    
    But the factorization convention is:
    U has shape (n*m, rank), indexed by (i*m + j, c)
    V has shape (m*p, rank), indexed by (j*p + k, c)  
    W has shape (p*n, rank), indexed by (k*n + i, c)
    
    And the reconstruction is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    where T[i,j,k] = 1 for the matmul tensor (since C[i,k] = sum_j A[i,j]B[j,k], each term A[i,j]B[j,k]
    contributes to C[i,k], so the tensor has T[i,j,k]=1).
    
    Wait, let me re-read. The output factor follows W[k*n+i, r].
    So W is indexed by (k*n + i) where k is the column of B (output column) and i is the row of A (output row).
    
    The tensor T[i,j,k] = 1 for all valid i,j,k (since matmul sums over j).
    """
    T = np.ones((n, m, p))
    return T

def _reconstruct(U, V, W, n, m, p):
    """Reconstruct the tensor from factors and compare with matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # W is indexed by (k*n + i), so W3[k, i, c]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
    return T_recon

def _error(U, V, W, n, m, p):
    """Compute the error between reconstructed tensor and matmul tensor."""
    T_recon = _reconstruct(U, V, W, n, m, p)
    T_target = np.ones((n, m, p))
    return np.sum((T_recon - T_target) ** 2)

def _random_init(n, m, p, rank, seed=0):
    """Random initialization of factors."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    return U, V, W

def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iter=5000, tol=1e-10):
    """Simple gradient descent to minimize error."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    for it in range(max_iter):
        T_recon = _reconstruct(U, V, W, n, m, p)
        T_target = np.ones((n, m, p))
        diff = T_recon - T_target  # (n, m, p)
        
        if np.sum(diff ** 2) < tol:
            break
        
        # Gradients
        # dT/dU[i,j,c] = sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        dU[i * m + j, c] += diff[i, j, k] * V3[j, k, c] * W3[k, i, c]
                        dV[j * p + k, c] += diff[i, j, k] * U3[i, j, c] * W3[k, i, c]
                        dW[k * n + i, c] += diff[i, j, k] * U3[i, j, c] * V3[j, k, c]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W

def _try_rank(n, m, p, rank, max_attempts=20, time_budget=10):
    """Try to find a valid rank-r decomposition."""
    start = time.time()
    best_err = float('inf')
    best_factors = None
    
    for attempt in range(max_attempts):
        if time.time() - start > time_budget:
            break
        
        seed = random.randint(0, 1000000)
        U, V, W = _random_init(n, m, p, rank, seed)
        U, V, W = _gradient_descent(U, V, W, n, m, p, lr=0.05, max_iter=2000, tol=1e-12)
        
        err = _error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-8:
            break
    
    if best_err < 1e-8 and best_factors is not None:
        U, V, W = best_factors
        # Convert to list format
        U_list = [[U[i, c] for c in range(rank)] for i in range(n * m)]
        V_list = [[V[i, c] for c in range(rank)] for i in range(m * p)]
        W_list = [[W[i, c] for c in range(rank)] for i in range(p * n)]
        return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try ranks from 10 upwards (known lower bound is 10 for 2x2x3 matmul)
    for rank in range(10, 13):
        if time.time() > deadline:
            break
        
        result = _try_rank(n, m, p, rank, max_attempts=30, time_budget=30)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 12
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
    return {"U": U, "V": V, "W": W}
