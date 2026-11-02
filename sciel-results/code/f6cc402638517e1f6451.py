"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time as _time

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly satisfies the tensor equations."""
    rank = len(U[0])
    # For each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _is_half_int(x):
    """Check if x is integer or half-integer (i.e., 2x is integer)."""
    return abs(2*x - round(2*x)) < 1e-9

def _to_half_int(x):
    """Convert to nearest half-integer."""
    return round(2*x) / 2.0

def construct():
    n, m, p = DIMENSIONS
    deadline = _time.time() + 140  # safety margin before 150s kill
    
    # Try to find a lower-rank decomposition
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Try ranks from a reasonable lower bound up to schoolbook
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # For small matrix multiplication, known results suggest rank can be much lower
    # Let's try a systematic search for ranks below schoolbook
    
    # Strategy: For each candidate rank, try to find a solution using
    # random initialization and iterative refinement
    
    for candidate_rank in range(28, best_rank):
        if _time.time() > deadline:
            break
        
        found = False
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if _time.time() > deadline:
                break
            
            # Random half-integer initialization in a reasonable range
            scale = 2
            U = [[random.randint(-scale, scale) / 2.0 for _ in range(candidate_rank)] for _ in range(n*m)]
            V = [[random.randint(-scale, scale) / 2.0 for _ in range(candidate_rank)] for _ in range(m*p)]
            W = [[random.randint(-scale, scale) / 2.0 for _ in range(candidate_rank)] for _ in range(p*n)]
            
            # Check if this random assignment works (very unlikely but possible for small ranks)
            if _verify(U, V, W, n, m, p):
                found = True
                best_rank = candidate_rank
                best_result = {"U": U, "V": V, "W": W}
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
