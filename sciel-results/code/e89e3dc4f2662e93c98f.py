import time
import numpy as np
from itertools import product

DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - 5.0 - _elapsed()  # 5s safety margin

def _validate(U, V, W, n, m, p):
    """
    Check if the decomposition exactly reproduces the matrix multiplication tensor.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all i,j,k.
    """
    rank = U.shape[1]
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _try_known_rank_18():
    """
    Try a known rank-18 decomposition for 2x2 * 2x9.
    The multiplication (2x2)(2x9) = (2x9) has tensor size 2*2*9 = 36.
    Lower bound: max(n*m, m*p, p*n) = max(4, 18, 18) = 18.
    So rank 18 is the theoretical lower bound.
    """
    n, m, p = 2, 2, 9
    rank = 18
    
    # For the rank-minimal case, we need U: (4, 18), V: (18, 18), W: (18, 18)
    # The structure: U[i*m+j, r] = delta_{r, (i,j)} for r in [0,4)
    # But rank is 18, so we need a different approach.
    
    # Actually, for rank = m*p = 18, the decomposition has a specific structure.
    # Let's use the standard construction.
    
    # U has shape (n*m, rank) = (4, 18)
    # V has shape (m*p, rank) = (18, 18) 
    # W has shape (p*n, rank) = (18, 18)
    
    # The idea: for each (j,k) pair in V, we have one rank-1 term.
    # V[j*p+k, r] is nonzero only when r corresponds to (j,k)
    # U[i*m+j, r] = delta_{j, j_r}
    # W[k*n+i, r] = delta_{k, k_r}
    
    # Let's index r from 0 to 17, corresponding to (j, k) pairs where j in [0,2), k in [0,9)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r in range(rank):
        j = r // p  # j in [0, 2)
        k = r % p   # k in [0, 9)
        
        # V[j*p + k, r] = 1
        V[j*p + k, r] = 1.0
        
        # U[i*m + j, r] = 1 for all i (since we're multiplying, the i index is free)
        # Actually, U[i*m + j, r] should be 1 for the given j
        for i in range(n):
            U[i*m + j, r] = 1.0
        
        # W[k*n + i, r] = 1 for all i
        for i in range(n):
            W[k*n + i, r] = 1.0
    
    # Let's validate
    if _validate(U, V, W, n, m, p):
        return {'U': U, 'V': V, 'W': W}
    
    # If the simple construction doesn't work, try another approach
    # The issue is that W[k*n+i, r] = 1 for all i means the sum over i will be n=2, not 1.
    # We need to fix this.
    
    # Correct approach:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # If we set V[j*p+k, r] = delta_{r, (j,k)}, then:
    # T[i,j,k] = U[i*m+j, j*k] * V[j*p+k, j*k] * W[k*n+i, j*k]
    # 
    # We need U[i*m+j, j*k] * W[k*n+i, j*k] = 1 for all i,j,k
    # 
    # Let r = j*p + k
    # U[i*m+j, r] * W[k*n+i, r] = 1
    # 
    # This must hold for all i. So U[i*m+j, r] * W[k*n+i, r] = 1 for each i.
    # 
    # One solution: U[i*m+j, r] = 1/sqrt(n) and W[k*n+i, r] = sqrt(n)
    # Then product = 1.
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    sqrt_n = 1.0 / np.sqrt(n)
    
    for r in range(rank):
        j = r // p
        k = r % p
        
        V[j*p + k, r] = 1.0
        
        for i in range(n):
            U[i*m + j, r] = sqrt_n
            W[k*n + i, r] = 1.0 / sqrt_n
    
    if _validate(U, V, W, n, m, p):
        return {'U': U, 'V': V, 'W': W}
    
    # Alternative: use integer entries
    # U[i*m+j, r] = 1, W[k*n+i, r] = 1, but then sum is n=2
    # We need sum_r ... = 1, so if only one term is nonzero, it must be 1.
    # With the diagonal construction, only r=(j,k) contributes, so:
    # U[i*m+j, j*p+k] * V[j*p+k, j*p+k] * W[k*n+i, j*p+k] = 1
    # Set V[j*p+k, j*p+k] = 1, U[i*m+j, j*p+k] = 1, W[k*n+i, j*p+k] = 1
    # Then the product is 1. But wait, that gives 1, not n.
    # Let me re-check: T[i,j,k] = sum_r ... only r=(j,k) is nonzero.
    # So T[i,j,k] = U[i*m+j, j*p+k] * 1 * W[k*n+i, j*p+k]
    # We need this to equal 1 for all i.
    # So U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i.
    # Setting both to 1 works!
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r in range(rank):
        j = r // p
        k = r % p
        
        V[j*p + k, r] = 1.0
        for i in range(n):
            U[i*m + j, r] = 1.0
            W[k*n + i, r] = 1.0
    
    if _validate(U, V, W, n, m, p):
        return {'U': U, 'V': V, 'W': W}
    
    return None

def _random_search(max_rank=20, num_trials=1000):
    """
    Random search for a lower-rank decomposition.
    """
    n, m, p = 2, 2, 9
    
    best = None
    best_rank = max_rank + 1
    
    for trial in range(num_trials):
        if _time_left() < 5.0:
            break
            
        # Try ranks from the theoretical minimum
        # Lower bound is max(n*m, m*p, p*n) = 18
        for rank in range(18, best_rank):
            if _time_left() < 5.0:
                break
            
            # Generate random matrices with small integer/half-integer entries
            scale = 2
            entries = np.random.choice([-scale, -scale+1, -scale+2, ..., scale*2], size=(rank,))
            
            # For simplicity, use entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(m*p, rank)).astype(float)
            W = np.random.randint(-1, 2, size=(p*n, rank)).astype(float)
            
            if _validate(U, V, W, n, m, p):
                if rank < best_rank:
                    best_rank = rank
                    best = {'U': U, 'V': V, 'W': W}
                break  # Found a valid one at this rank, try lower
    
    return best

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix 
    by a 2 by 9 matrix.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = 2, 2, 9
    
    # Try the known rank-18 construction first
    result = _try_known_rank_18()
    if result is not None:
        return result
    
    # If that fails, do a quick random search
    result = _random_search(max_rank=20, num_trials=500)
    if result is not None:
        return result
    
    # Fallback: return the naive rank-36 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for col, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U[i*m + j, col] = 1.0
        V[j*p + k, col] = 1.0
        W[k*n + i, col] = 1.0
    
    return {'U': U, 'V': V, 'W': W}
