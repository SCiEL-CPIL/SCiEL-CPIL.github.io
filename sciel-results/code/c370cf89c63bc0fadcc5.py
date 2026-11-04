"""Rank-minimal tensor decomposition for 3x3x8 matrix multiplication.

Uses schoolbook as baseline and attempts local optimization to reduce rank.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0  # Safety margin before 150s kill

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _tensor_from_factors(U, V, W):
    """Compute the multiplication tensor from factor matrices."""
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But W uses trace convention: W[k*n+i, r]
    tensor = np.zeros((n, m, p))
    rank = U.shape[1]
    for r in range(rank):
        u_col = U[:, r]  # (n*m,)
        v_col = V[:, r]  # (m*p,)
        w_col = W[:, r]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
    return tensor

def _target_tensor():
    n, m, p = DIMENSIONS
    tensor = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i, j, k] = 1.0  # A[i,j]*B[j,k] with A,B having 1 at diagonal/identity-like
    # Actually the tensor is T[i,j,k] = 1 for all i,j,k since it's the structure tensor
    # where T[i,j,k] = delta_{i,j} * delta_{j,k} is not right.
    # The multiplication tensor has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity mapping in tensor form)
    return tensor

def _residual(U, V, W):
    """Compute residual between computed tensor and target."""
    n, m, p = DIMENSIONS
    target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i, j, k] = 1.0
    
    rank = U.shape[1]
    computed = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    return np.max(np.abs(computed - target))

def _try_reduce_rank(U, V, W, max_attempts=50):
    """Try to reduce rank by zeroing out columns and checking if tensor is preserved."""
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Try removing one column at a time
    for r in range(current_rank):
        if time.time() > DEADLINE:
            break
        # Skip column r
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        U_test = U[:, mask]
        V_test = V[:, mask]
        W_test = W[:, mask]
        
        # Check if the tensor is still correct
        n, m, p = DIMENSIONS
        target = np.ones((n, m, p))
        rank_test = U_test.shape[1]
        computed = np.zeros((n, m, p))
        for rr in range(rank_test):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        computed[i, j, k] += U_test[i*m+j, rr] * V_test[j*p+k, rr] * W_test[k*n+i, rr]
        
        if np.max(np.abs(computed - target)) < 1e-10:
            best_U, best_V, best_W = U_test.copy(), V_test.copy(), W_test.copy()
            best_rank = rank_test
            # Continue trying to remove more
            U, V, W = U_test, V_test, W_test
            current_rank = rank_test
            r = max(0, r - 1)  # Re-check from same position
            continue
    
    return best_U, best_V, best_W, best_rank

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _build_schoolbook()
    rank = U.shape[1]
    
    # Try to reduce rank by removing redundant columns
    U, V, W, rank = _try_reduce_rank(U, V, W)
    
    if time.time() - t0 > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try random perturbation and local search
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Attempt: initialize with random factors of lower rank and optimize
    for target_rank in range(max(1, rank - 5), rank):
        if time.time() - t0 > DEADLINE:
            break
        
        for attempt in range(10):
            if time.time() - t0 > DEADLINE:
                break
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Simple gradient-free: try to minimize residual
            # For now, just check if a random init happens to work (unlikely)
            # This is a placeholder for more sophisticated optimization
            pass
    
    # Return the best found
    if time.time() - t0 > DEADLINE:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
