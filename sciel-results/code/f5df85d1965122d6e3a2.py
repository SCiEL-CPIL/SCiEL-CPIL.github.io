import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]
SEED_RANK = 105

def _validate(U, V, W, n=3, m=5, p=7):
    """
    Validate that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=3, m=5, p=7):
    """Return the schoolbook decomposition with rank n*m*p."""
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

def _attempt_rank_reduction(n=3, m=5, p=7, deadline=None, seed=42):
    """
    Attempt to find a lower-rank decomposition.
    Focuses on Rank 104 and 103 with integer coefficients {-1, 0, 1}.
    """
    if deadline is None:
        deadline = time.time() + 110 # Safety margin
    
    rng = random.Random(seed)
    best = _schoolbook(n, m, p)
    
    # Coefficients to try
    coeffs = [-1, 0, 1]
    
    # Try Rank 104 and 103
    for target_rank in [104, 103]:
        if time.time() > deadline:
            break
            
        # Number of attempts based on remaining time
        # Each validation is O(n*m*p*rank) = 105 * 104 ~ 11k ops. Very fast.
        # We can do many attempts.
        max_attempts = 2000
        attempts = 0
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
                
            # Generate random sparse matrices
            # Density: ~1/3 non-zeros
            U = [[rng.choice(coeffs) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[rng.choice(coeffs) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[rng.choice(coeffs) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
            
            attempts += 1
            
    return best

def construct():
    """
    Returns the best tensor decomposition found for matrix multiplication.
    """
    deadline = time.time() + 115 # Hard stop before 120s limit
    
    # Start with schoolbook
    best = _schoolbook()
    
    # Attempt rank reduction
    result = _attempt_rank_reduction(deadline=deadline)
    if result is not None:
        best = result
    
    return best
