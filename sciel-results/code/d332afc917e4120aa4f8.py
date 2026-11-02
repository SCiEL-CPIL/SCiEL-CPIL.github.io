"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
DEADLINE = 140.0  # 10s safety margin

# EVOLVE-BLOCK-START
def _time_remaining():
    return DEADLINE - time.time()

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    # Check shapes
    if len(U) != n * m or len(U[0]) != rank:
        return False
    if len(V) != m * p or len(V[0]) != rank:
        return False
    if len(W) != p * n or len(W[0]) != rank:
        return False
    
    # Verify by computing the reconstructed tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-n*m*p decomposition."""
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

def _try_strassen_like(n, m, p):
    """
    Try to find a low-rank decomposition using a Strassen-like approach.
    
    For 3x4x5, we can try to block the matrices and use a Strassen-like
    multiplication on the blocks. The key idea is to find a decomposition
    with fewer than n*m*p = 60 terms.
    
    We'll use a parameterized family based on the algebraic structure of
    matrix multiplication.
    """
    start_time = time.time()
    
    # Strategy: Use the fact that we can potentially reduce rank by exploiting
    # the rectangular structure. For 3x4x5, we can try:
    # 1. Block A as (2x4) + (1x4), B as (4x3) + (4x2)
    # 2. Use Strassen-like formulas on the blocks
    
    # Let's try a direct search over small integer/half-integer values
    # with a structured approach.
    
    # The tensor equation is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # We'll try to find a rank-59 decomposition by starting from schoolbook
    # and trying to merge terms.
    
    # Approach: Use random linear combinations of schoolbook terms
    # and check if we can reduce the rank.
    
    # For efficiency, let's try a specific known construction.
    # The 3x4x5 tensor might have rank 59 based on known results.
    
    # Let's try a greedy approach: start with schoolbook and try to
    # replace pairs of terms with a single term.
    
    U_school = _build_schoolbook(n, m, p)["U"]
    V_school = _build_schoolbook(n, m, p)["V"]
    W_school = _build_schoolbook(n, m, p)["W"]
    
    # Try to find if there's a linear dependency among the 60 rank-1 terms
    # that allows us to reduce the rank.
    
    # The rank-1 terms are:
    # Term (i,j,k): U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    
    # We need to find coefficients alpha_{i,j,k} such that:
    # sum_{i,j,k} alpha_{i,j,k} * (e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}) = 0
    # and at least one alpha is nonzero, which would mean the terms are linearly dependent.
    
    # But this is a 3-tensor, and linear dependence in the tensor sense is complex.
    
    # Alternative approach: Use the "border rank" or "approximate rank" ideas
    # but we need exact decomposition.
    
    # Let's try a different strategy: use the fact that for some rectangular
    # matrix multiplications, we can achieve sub-schoolbook ranks.
    
    # For 3x4x5, let's try to construct a rank-59 decomposition by
    # finding a valid set of 59 terms.
    
    # We'll use a randomized local search: start with schoolbook, then
    # try to perturb and see if we can maintain validity with fewer terms.
    
    # Actually, let's try a more systematic approach:
    # Assume a decomposition of the form:
    # U[i*m+j,c] = sum over some basis
    # This is getting complex. Let's try a simple random search with
    # verification.
    
    # Generate random decompositions and check validity.
    # For rank r, we need r columns in U, V, W.
    
    # Let's try ranks from 59 down to 50 and see if we can find a valid one.
    
    import random
    random.seed(42)
    
    for target_rank in range(59, 49, -1):
        if time.time() - start_time > 30:
            break
        
        # Try multiple random initializations
        for attempt in range(100):
            if time.time() - start_time > 30:
                break
            
            # Initialize with random values in {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify_decomposition(U, V, W, n, m, p, target_rank):
                return {"U": U, "V": V, "W": W}
        
        # Try with half-integers
        for attempt in range(50):
            if time.time() - start_time > 30:
                break
            
            values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify_decomposition(U, V, W, n, m, p, target_rank):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First try to find a low-rank decomposition
    result = _try_strassen_like(n, m, p)
    if result is not None:
        return result
    
    # Fall back to schoolbook rank 60
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
