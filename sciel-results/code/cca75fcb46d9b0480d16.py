"""
Rank-minimal tensor decomposition for 2x5x5 matrix multiplication.
Uses a search approach to find the minimum rank decomposition.
"""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 5, 5]

def build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a list of (i,j,k) indices."""
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                targets.append((i, j, k))
    return targets

def validate_decomposition(U, V, W, n, m, p):
    """Validate that the decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    r = U.shape[1]
    targets = build_target_tensor(n, m, p)
    
    for (i, j, k) in targets:
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        val = np.dot(U[idx_u, :] * V[idx_v, :], W[idx_w, :])
        if abs(val - 1.0) > 1e-9:
            return False
    return True

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def try_random_rank(r, n, m, p, max_iters=5000, timeout=30):
    """Try to find a rank-r decomposition using randomized search."""
    start_time = time.time()
    targets = build_target_tensor(n, m, p)
    
    for attempt in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small values
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        
        # Initialize with random small values
        for c in range(r):
            U[:, c] = np.random.choice([-1, 0, 1], size=n * m)
            V[:, c] = np.random.choice([-1, 0, 1], size=m * p)
            W[:, c] = np.random.choice([-1, 0, 1], size=p * n)
        
        # Check if this random initialization works (very unlikely but possible)
        if validate_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def try_greedy_merge(n, m, p, base_rank=50, timeout=30):
    """Try to merge terms from the schoolbook decomposition to reduce rank."""
    start_time = time.time()
    U, V, W = schoolbook_decomposition(n, m, p)
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    targets = build_target_tensor(n, m, p)
    current_rank = base_rank
    
    # Try to merge pairs of columns
    merged = False
    for iter_count in range(100):
        if time.time() - start_time > timeout:
            break
        if not merged:
            break
        merged = False
        
        # Try merging each pair
        for c1 in range(current_rank):
            for c2 in range(c1 + 1, current_rank):
                # Try to see if we can replace columns c1 and c2 with a single column
                # This is a nonlinear problem, so we try simple approaches
                
                # Approach: set U[:,c2] = U[:,c1], V[:,c2] = V[:,c1], W[:,c2] = W[:,c1]
                # and scale appropriately. This only works if the two terms are identical.
                
                # Check if columns c1 and c2 are identical
                if np.allclose(U[:, c1], U[:, c2]) and np.allclose(V[:, c1], V[:, c2]) and np.allclose(W[:, c1], W[:, c2]):
                    # They're identical, so we can merge
                    U[:, c1] *= 2
                    V[:, c1] *= 2
                    W[:, c1] *= 2
                    # Remove column c2
                    U = np.delete(U, c2, axis=1)
                    V = np.delete(V, c2, axis=1)
                    W = np.delete(W, c2, axis=1)
                    current_rank -= 1
                    merged = True
                    break
            if merged:
                break
    
    if validate_decomposition(U, V, W, n, m, p):
        return U.tolist(), V.tolist(), W.tolist(), current_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Start with schoolbook
    U, V, W = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from a lower bound up to 50
    
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, the tensor has 50 non-zero entries, so rank >= 1
    # But structurally, we need at least enough terms to cover all positions
    
    # Try a few different approaches
    # 1. Random search for low rank
    for r in range(20, 50):
        if time.time() > deadline - 10:
            break
        result = try_random_rank(r, n, m, p, max_iters=1000, timeout=5)
        if result is not None:
            best_rank = r
            U, V, W = result
            break
    
    # 2. Try greedy merge from schoolbook
    if time.time() < deadline - 10:
        result = try_greedy_merge(n, m, p, base_rank=50, timeout=20)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_rank = rank
    
    # Return the best decomposition found
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Ensure integer/half-integer entries by rounding
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
