import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k] = delta_{i,k} * delta_{j,i} (conceptually).
    Actually, the tensor T has entries T[(i,j),(j,k),(k,i)] = 1 for the multiplication.
    We represent it as a 3D array where T[a,b,c] = 1 if a corresponds to (i,j), b to (j,k), c to (k,i).
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
    
    # Build the true tensor
    T_true = _build_tensor(n, m, p)
    
    # Check exact match (with tolerance for floating point)
    return np.allclose(T_decomp, T_true, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _random_decomposition(n, m, p, rank, rng):
    """Generate a random decomposition with integer/half-integer entries."""
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Use small integers: -1, 0, 1
    for r in range(rank):
        for i in range(n*m):
            if rng.random() < 0.3:
                U[i, r] = rng.choice([-1, 1])
        for j in range(m*p):
            if rng.random() < 0.3:
                V[j, r] = rng.choice([-1, 1])
        for k in range(p*n):
            if rng.random() < 0.3:
                W[k, r] = rng.choice([-1, 1])
    
    return U, V, W

def _try_rank(n, m, p, rank, max_attempts, rng, deadline):
    """Try to find a decomposition of given rank."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        U, V, W = _random_decomposition(n, m, p, rank, rng)
        
        if _verify(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for matrix multiplication of sizes 2x3 and 3x7.
    Uses search to find lower-rank decompositions if possible, falls back to schoolbook (rank 42).
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Set a deadline with safety margin
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, leave margin
    
    # Start with schoolbook as the best known
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    rng = np.random.default_rng(42)
    
    # Try ranks from low to high
    # For 2x3x7, lower bounds suggest rank >= 14 (Landsberg bound) but practical lower bound is higher
    # Try ranks from 14 up to schoolbook_rank - 1
    for rank in range(14, best_rank):
        if time.time() > deadline - 10:  # Leave 10 seconds margin
            break
        
        # Number of attempts based on remaining time
        remaining_time = deadline - time.time()
        max_attempts = max(10, int(remaining_time * 10))  # 10 attempts per second
        
        result = _try_rank(n, m, p, rank, max_attempts, rng, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition at this rank, we're done
    
    return best_result
# EVOLVE-BLOCK-END
