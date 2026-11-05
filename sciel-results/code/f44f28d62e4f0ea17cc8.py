"""Rewritten program for matrix multiplication 3,3,8 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the multiplication tensor."""
    # Compute the reconstructed tensor
    # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_recon, T_true, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _kronecker_decomposition_448():
    """
    Build the rank-16 decomposition of M_{4,4,8} using Kronecker products of M_{2,2,2}.
    
    M_{2,2,2} has a rank-4 decomposition:
    U2, V2, W2 with shapes (4,4), (4,4), (4,4)
    
    The Kronecker product gives M_{4,4,8} with rank 4*4=16.
    """
    # M_{2,2,2} schoolbook decomposition (rank 8)
    # But we can do better: rank 4 for 2x2x2
    # Actually, 2x2x2 matrix multiplication has rank 7 (Strassen), but let's use schoolbook first
    # For simplicity, use the known rank-4 decomposition if available, else schoolbook
    
    # Schoolbook for 2x2x2: rank 8
    U2 = np.zeros((4, 8))
    V2 = np.zeros((4, 8))
    W2 = np.zeros((4, 8))
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U2[i*2 + j, col] = 1
                V2[j*2 + k, col] = 1
                W2[k*2 + i, col] = 1
                col += 1
    
    # Kronecker product to get 4x4x8
    # U4[a, r] = U2[a1, r1] * U2[a2, r2] where a = a1*2 + a2, r = r1*4 + r2
    # This gives rank 8*8 = 64, which is too high
    
    # Let me use the actual rank-7 Strassen decomposition for 2x2x2
    # Then Kronecker gives rank 49 for 4x4x8, still too high
    
    # Actually, let me just return the schoolbook for now and try a different approach
    return None

def _numerical_optimization(n, m, p, initial_rank, time_limit=100.0):
    """
    Try to find a lower rank decomposition using numerical optimization.
    Start from schoolbook and try to reduce rank by solving for a lower-rank approximation.
    """
    start_time = time.time()
    
    # Try ranks from 40 down to 20 (reasonable range for 3x3x8)
    for target_rank in range(40, 19, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random or structured values
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Use gradient descent to minimize ||T_recon - T_true||^2
        T_true = _build_tensor(n, m, p)
        
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_recon - T_true
            
            # Gradients
            grad_U = np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = np.einsum('abc,ar,br->cr', error, U, V)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.linalg.norm(error) < 1e-8:
                if _verify(U, V, W, n, m, p, tol=1e-7):
                    return U, V, W, target_rank
        
        # Final check
        if _verify(U, V, W, n, m, p, tol=1e-7):
            return U, V, W, target_rank
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try numerical optimization to reduce rank
    if time.time() < deadline - 10:
        U2, V2, W2, rank2 = _numerical_optimization(n, m, p, best_rank, time_limit=90.0)
        if rank2 is not None and rank2 < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = rank2
    
    # Convert to lists
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
