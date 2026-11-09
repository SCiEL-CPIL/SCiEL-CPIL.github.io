"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    
    # For each output element (i, k) of the n x p result matrix,
    # we need: sum_r U[i*m + j][r] * V[j*p + k][r] = 1 if j exists, else 0
    # Actually the tensor T[i,j,k] = delta(i,j,k) for matrix mult
    # T[i,j,k] = 1 if we're computing element (i,k) using row i of A and col k of B
    # The decomposition requires: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    
    # Let's verify by checking the tensor contraction
    # T[i,j,k] should equal delta_{i,k} * 1 (the matrix multiplication tensor)
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the standard matmul tensor)
    
    # Check: for each (i, j, k), sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _random_search(n, m, p, max_rank, time_budget=140.0, seed=None):
    """Random search for a low-rank exact decomposition."""
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    best_decomp = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try ranks from some lower bound up to max_rank
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually rank must be at least max(n*m, m*p, p*n) in general for exact
    # But for matmul, the lower bound is more subtle. Let's try a range.
    
    min_rank = max(n * m, m * p, p * n)  # crude lower bound
    
    for rank in range(min_rank, max_rank + 1):
        if time.time() - start_time > time_budget:
            break
        
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts:
            if time.time() - start_time > time_budget:
                break
            
            attempts += 1
            
            # Generate random U, V, W with small integer/half-integer entries
            # Use values from {-1, 0, 1, 0.5, -0.5}
            vals = [-1, -0.5, 0, 0.5, 1]
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for _ in range(n * m * rank):
                U.flat[_] = random.choice(vals)
            for _ in range(m * p * rank):
                V.flat[_] = random.choice(vals)
            for _ in range(p * n * rank):
                W.flat[_] = random.choice(vals)
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(val - 1.0) > 1e-10:
                            valid = False
                            break
            
            if valid:
                best_rank = rank
                best_decomp = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break  # Found for this rank, move to next lower... wait, we're going up
    
    if best_decomp is not None:
        return best_decomp
    return None

def construct():
    # We construct an exact decomposition for 3x4 * 4x8 -> 3x8.
    # n=3, m=4, p=8.
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # First, get the schoolbook solution as fallback
    fallback = _schoolbook_decomp(n, m, p)
    
    # Try random search for better decomposition
    # Use a time budget of 135 seconds to leave safety margin
    result = _random_search(n, m, p, max_rank=96, time_budget=135.0, seed=42)
    
    if result is not None:
        return result
    
    # Check if we're still within time limit
    elapsed = time.time() - start_time
    if elapsed > 145.0:
        return fallback
    
    # Return the best we have
    return fallback
# EVOLVE-BLOCK-END
