"""Program that attempts to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import time
import sys
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]
SEED_RANK = 105

# EVOLVE-BLOCK-START

def _validate(U, V, W, n=3, m=5, p=7):
    """
    Validate that the factor matrices U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The condition is: for all i, j, k,
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
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


def _try_local_search(U, V, W, n, m, p, deadline, rng, max_steps=1000):
    """
    Start from a valid decomposition and try to reduce rank by zeroing out terms
    and adjusting others to compensate.
    """
    rank = len(U[0])
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Try to eliminate one term at a time
        for r in range(rank):
            if time.time() > deadline:
                break
            
            # Save current state
            U_save = [row[:] for row in U]
            V_save = [row[:] for row in V]
            W_save = [row[:] for row in W]
            
            # Zero out term r
            for i in range(n * m):
                U[i][r] = 0
            for j in range(m * p):
                V[j][r] = 0
            for k in range(p * n):
                W[k][r] = 0
            
            # Check if still valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for rr in range(rank):
                            s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Try to remove the column entirely
                U = [row[:r] + row[r+1:] for row in U]
                V = [row[:r] + row[r+1:] for row in V]
                W = [row[:r] + row[r+1:] for row in W]
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                break
            else:
                # Restore
                U, V, W = U_save, V_save, W_save
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def _attempt_rank_reduction(n=3, m=5, p=7, deadline=None, seed=42):
    """
    Attempt to find a lower-rank decomposition using structured approaches.
    """
    if deadline is None:
        deadline = time.time() + 140
    
    rng = random.Random(seed)
    best = _schoolbook(n, m, p)
    
    # Start with schoolbook and try local search
    result = _try_local_search(best["U"], best["V"], best["W"], n, m, p, deadline, rng)
    if result is not None:
        best = result
    
    # Try random structured approaches for lower ranks
    def validate2(U, V, W, n=3, m=5, p=7):
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
    
    return best


def construct():
    """
    Returns the best tensor decomposition found for matrix multiplication.
    """
    deadline = time.time() + 140
    
    # First, return schoolbook as a safe fallback
    best = _schoolbook()
    
    # Attempt rank reduction
    if time.time() < deadline:
        result = _attempt_rank_reduction(deadline=deadline)
        if result is not None:
            # Validate the result before accepting it
            if _validate(result["U"], result["V"], result["W"]):
                best = result
            else:
                # If validation fails, stick with schoolbook
                best = _schoolbook()
    
    return best

# EVOLVE-BLOCK-END
