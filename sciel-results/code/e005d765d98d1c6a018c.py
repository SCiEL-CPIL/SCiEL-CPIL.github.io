"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, rank, n, m, p):
    """Check if the decomposition is valid."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_strassen_block(n, m, p):
    """Try to use Strassen-like decomposition on blocks."""
    # For 2x2x2, Strassen uses 7 multiplications
    # We can try to apply this to sub-blocks of our 3x5x7 problem
    
    # One approach: split the middle dimension (5) into parts
    # and use Strassen on 2x2 sub-blocks where possible
    
    # Let's try a different approach: use a known low-rank decomposition
    # by searching for one
    
    return None

def _random_search(n, m, p, max_rank, deadline, seed=42):
    """Try random search for a low-rank decomposition."""
    random.seed(seed)
    np.random.seed(seed)
    
    best = None
    best_rank = max_rank + 1
    
    for target_rank in range(60, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with small random values
            U = np.random.randint(-2, 3, (n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, (m*p, target_rank)).astype(float)
            W = np.random.randint(-2, 3, (p*n, target_rank)).astype(float)
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best is not None:
            break
    
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook as fallback
    schoolbook_result = _schoolbook(n, m, p)
    best_result = schoolbook_result
    best_rank = n * m * p  # 105
    
    # Try to find a better decomposition
    # Strategy: use structured decomposition based on bilinear forms
    
    # For 3x5x7, let's try to construct a decomposition manually
    # using the fact that we can share computations
    
    # One key insight: if we have terms that differ only in one index,
    # we might be able to combine them
    
    # Let's try a systematic approach: build up the decomposition
    # by adding rank-1 terms that cover multiple (i,j,k) positions
    
    # Initialize with zeros
    U = [[0] * 105 for _ in range(n * m)]
    V = [[0] * 105 for _ in range(m * p)]
    W = [[0] * 105 for _ in range(p * n)]
    
    # Track which (i,j,k) positions are covered
    covered = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered[(i,j,k)] = 0.0
    
    # Try to add rank-1 terms that cover multiple positions
    # A rank-1 term with U having nonzeros at (i1,j1), (i2,j2), etc.
    # V having nonzeros at (j1,k1), (j2,k2), etc.
    # W having nonzeros at (k1,i1), (k2,i2), etc.
    # contributes to all combinations
    
    # This is complex, so let's try a different approach:
    # Use a known low-rank decomposition if available
    
    # For now, let's try a random search with a focus on sparse solutions
    result, rank = _random_search(n, m, p, 105, deadline, seed=random.randint(0, 10000))
    
    if result is not None and rank < best_rank:
        best_result = result
        best_rank = rank
    
    # If we still have time, try more sophisticated approaches
    if time.time() < deadline - 30:
        # Try a gradient-based approach
        pass
    
    return best_result

# EVOLVE-BLOCK-END
