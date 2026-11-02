"""Rank-minimal exact tensor decomposition search for 3x5x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _strassen_3x3():
    """Return Strassen decomposition for 3x3x3 tensor, rank 7."""
    # Strassen's 7-multiplication algorithm for 3x3 matrices
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using standard Strassen formulas adapted to tensor form
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]
    
    # Strassen's 7 products for 3x3:
    # M1 = (A11+A33)*(B11+B33)
    # M2 = (A22+A33)*(B22+B33)
    # M3 = (A11+A22)*(B11+B22)
    # M4 = (A33+A11)*(B33+B11)
    # M5 = (A11-A22)*(B21-B11)
    # M6 = (A22-A33)*(B32-B22)
    # M7 = (A33-A11)*(B13-B33)
    
    # This is complex; let me use a known valid rank-7 decomposition
    # For 3x3x3, one valid decomposition:
    U_data = [
        [1,0,0,0,1,0,0],  # i=0,j=0
        [0,0,1,0,0,0,0],  # i=0,j=1
        [0,0,0,1,0,0,0],  # i=0,j=2
        [0,1,0,0,0,1,0],  # i=1,j=0
        [0,0,1,0,0,0,0],  # i=1,j=1
        [0,0,0,0,0,0,1],  # i=1,j=2
        [1,0,0,0,0,0,0],  # i=2,j=0
        [0,1,0,0,0,0,0],  # i=2,j=1
        [0,0,1,0,0,0,0],  # i=2,j=2
    ]
    V_data = [
        [1,0,0,0,0,0,1],  # j=0,k=0
        [0,1,0,0,0,0,0],  # j=0,k=1
        [0,0,1,0,0,0,0],  # j=0,k=2
        [0,1,0,0,1,0,0],  # j=1,k=0
        [0,0,1,0,0,1,0],  # j=1,k=1
        [0,0,0,1,0,0,0],  # j=1,k=2
        [1,0,0,0,0,0,0],  # j=2,k=0
        [0,1,0,0,0,0,0],  # j=2,k=1
        [0,0,1,0,0,0,1],  # j=2,k=2
    ]
    W_data = [
        [1,0,0,0,0,0,0],  # k=0,i=0
        [0,1,0,0,0,0,0],  # k=0,i=1
        [0,0,1,0,0,0,0],  # k=0,i=2
        [0,0,0,1,0,0,0],  # k=1,i=0
        [0,0,0,0,1,0,0],  # k=1,i=1
        [0,0,0,0,0,1,0],  # k=1,i=2
        [0,0,0,0,0,0,1],  # k=2,i=0
        [0,0,0,0,0,0,0],  # k=2,i=1
        [0,0,0,0,0,0,0],  # k=2,i=2
    ]
    # Verify this is correct - it may not be, so let's use a simpler approach
    # For now, return None if verification fails
    if _check_exact(U_data, V_data, W_data, 3, 3, 3):
        return U_data, V_data, W_data
    return None


def _build_block_decomposition(n, m, p, block_size, strassen_decomp):
    """
    Build a block-based decomposition using Strassen-like reductions.
    For 3x5x6, we can think of it as blocks and apply rank reductions.
    """
    if strassen_decomp is None:
        return None
    
    U_s, V_s, W_s = strassen_decomp
    rank_s = len(U_s[0])
    
    # For 3x5x6, we can decompose into 3x3 blocks with some remainder
    # This is complex; let's try a different approach
    return None


def _random_search(n, m, p, target_rank, seed, val_range=(-2, 2), max_iter=50):
    """Randomized search for a valid decomposition with given rank."""
    rng = random.Random(seed)
    for _ in range(max_iter):
        U = [[rng.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[rng.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(p * n)]
        if _check_exact(U, V, W, n, m, p):
            return U, V, W
    return None


def _structured_search(n, m, p, target_rank, deadline):
    """
    Structured search: try to build a decomposition with specific patterns.
    For 3x5x6, try block-based approaches.
    """
    # Try a few structured patterns
    patterns = []
    
    # Pattern 1: Diagonal + off-diagonal corrections
    # Pattern 2: Block diagonal with Strassen-like reductions
    
    # For now, do a more targeted random search with structured initialization
    rng = random.Random(42)
    
    # Initialize with identity-like structure
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Fill with small random values, but ensure some structure
        for idx in range(n * m):
            for c in range(target_rank):
                U[idx][c] = rng.randint(-1, 1)
        for idx in range(m * p):
            for c in range(target_rank):
                V[idx][c] = rng.randint(-1, 1)
        for idx in range(p * n):
            for c in range(target_rank):
                W[idx][c] = rng.randint(-1, 1)
        
        if _check_exact(U, V, W, n, m, p):
            return U, V, W
    
    return None


def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin under 150s
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (U0, V0, W0)
    best_rank = n * m * p
    
    # Try Strassen for 3x3x3 as a building block
    strassen_3x3 = _strassen_3x3()
    
    # Search for lower rank decompositions
    # Lower bound: at least max(n*m, m*p, p*n) = max(15,30,18) = 30
    # But tensor rank can be lower due to structure
    # For 3x3x3, rank is 7, so for 3x5x6, expect significantly less than 90
    
    # Try ranks from a reasonable lower bound up to schoolbook-1
    # Start with a focused range
    search_start = 25
    search_end = min(best_rank - 1, 60)  # Don't go too high initially
    
    for target_rank in range(search_start, search_end + 1):
        if time.time() > deadline:
            break
        
        # Try structured search first
        result = _structured_search(n, m, p, target_rank, deadline)
        if result is not None:
            best = result
            best_rank = target_rank
            break
        
        # Try random search with a few iterations
        result = _random_search(n, m, p, target_rank, seed=42 + target_rank, max_iter=10)
        if result is not None:
            best = result
            best_rank = target_rank
            break
    
    # If we found an improvement, try to go even lower
    if best_rank < n * m * p:
        for target_rank in range(max(20, best_rank - 5), best_rank):
            if time.time() > deadline:
                break
            result = _random_search(n, m, p, target_rank, seed=100 + target_rank, max_iter=20)
            if result is not None:
                best = result
                best_rank = target_rank
                break
    
    U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
