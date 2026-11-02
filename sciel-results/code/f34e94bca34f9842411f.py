"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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


def _validate(U, V, W, n, m, p):
    """Check that U, V, W exactly decompose the matmul tensor.
    
    The matmul tensor T has T[i, j, k] = 1 iff j == k (shared index),
    where i in [0,n), j in [0,m), k in [0,p).
    
    Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0])
    # Build tensor from factors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if j == k else 0
                if s != expected:
                    return False
    return True


def _validate_np(U, V, W, n, m, p):
    """Vectorized validation using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reshape: Ua[i*m+j, c] -> Ua3[i, j, c]
    U3 = Ua.reshape(n, m, rank)
    # Va[j*p+k, c] -> Va3[j, k, c]
    V3 = Va.reshape(m, p, rank)
    # Wa[k*n+i, c] -> Wa3[k, i, c]
    W3 = Wa.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Use einsum: U3[i,j,c] * V3[j,k,c] -> UV[i,j,k,c], then * W3[k,i,c]
    UV = np.einsum('ijc,jkc->ijkc', U3, V3)
    T = np.einsum('ijkc,kic->ijk', UV, W3)
    
    # Expected: T[i,j,k] = 1 if j==k else 0
    expected = np.zeros((n, m, p), dtype=np.float64)
    for j in range(min(m, p)):
        expected[:, j, j] = 1.0
    
    return np.allclose(T, expected, atol=1e-9)


def _random_decomposition(n, m, p, rank, rng):
    """Generate a random rank-r decomposition and try to refine it."""
    # Initialize with random values
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for c in range(rank):
        U[:, c] = rng.choice([-1, 0, 1], size=n * m)
        V[:, c] = rng.choice([-1, 0, 1], size=m * p)
        W[:, c] = rng.choice([-1, 0, 1], size=p * n)
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin: kill at 150s, leave 40s buffer
    
    # Start with schoolbook (rank 45)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Known minimal rank for (3,3,5) is 43
    # Try to find decompositions with rank < 45
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    
    # Try ranks from 43 downward
    for target_rank in range(43, 44, -1):  # Try 43 first
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Initialize with structured pattern: start from schoolbook and merge
            # For rank 43, we need to merge 2 of the 45 schoolbook terms
            # The key insight: for (3,3,5), some terms can be combined
            
            # Strategy: try to find a valid rank-43 decomposition
            # by perturbing a near-valid structure
            
            # Build a structured initialization based on schoolbook
            # with some entries merged
            for c in range(target_rank):
                # Assign random but structured values
                U[:, c] = np_rng.choice([-1, 0, 1], size=n * m)
                V[:, c] = np_rng.choice([-1, 0, 1], size=m * p)
                W[:, c] = np_rng.choice([-1, 0, 1], size=p * n)
            
            # Check if valid
            if _validate_np(U, V, W, n, m, p):
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < n * m * p:
        return best
    else:
        return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
