"""Rewritten program: attempt to find a lower-rank exact tensor decomposition for 2x3 by 3x7."""
import time
import random
import math
import sys
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _try_structured(rank, rng, deadline):
    """Try structured initializations that exploit tensor symmetries."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Strategy 1: Block diagonal structure based on j index
    # For each j, have a block of size ~rank/m
    block_size = rank // m
    remaining = rank % m
    
    for _ in range(200):
        if time.time() > deadline:
            return None
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        for j in range(m):
            bs = block_size + (1 if j < remaining else 0)
            for r in range(bs):
                # Randomly assign which i,k pairs get non-zero values
                for i in range(n):
                    if rng.random() < 0.7:
                        U[i * m + j][col + r] = rng.choice([-1, 1])
                for k in range(p):
                    if rng.random() < 0.7:
                        V[j * p + k][col + r] = rng.choice([-1, 1])
                    if rng.random() < 0.7:
                        W[k * n + i][col + r] = rng.choice([-1, 1])
            col += bs
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def _local_repair(U, V, W, max_iters=50, deadline=None):
    """Try to fix violations by flipping entries."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Find a violation
        violations = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        violations.append((i, j, k, s))
        
        if not violations:
            return {"U": U, "V": V, "W": W}
        
        # Pick a random violation
        i, j, k, s = random.choice(violations)
        
        # Try flipping an entry to fix it
        # The contribution of term r is U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        # We need sum = 1, currently sum = s
        # Flipping U[i*m+j][r] changes contribution by -2*U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]
        
        for r in range(rank):
            contrib = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            # If we flip U, new contrib = -contrib, change = -2*contrib
            # We want s - 2*contrib = 1, so contrib = (s-1)/2
            if (s - 1) % 2 == 0 and contrib == (s - 1) // 2:
                U[i * m + j][r] *= -1
                break
            # If we flip V
            if (s - 1) % 2 == 0 and contrib == (s - 1) // 2:
                V[j * p + k][r] *= -1
                break
            # If we flip W
            if (s - 1) % 2 == 0 and contrib == (s - 1) // 2:
                W[k * n + i][r] *= -1
                break
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0]) if best["U"] else 0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    rng = random.Random(42)
    
    # Try structured approaches first
    for rank in range(28, 20, -1):
        if time.time() > deadline:
            break
        result = _try_structured(rank, rng, deadline)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # If structured didn't work, try random with repair
    for rank in range(35, 20, -1):
        if time.time() > deadline:
            break
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization with sparsity
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            
            for i in range(nm):
                for r in range(rank):
                    if rng.random() < 0.5:
                        U[i][r] = rng.choice([-1, 1])
            
            for j in range(mp):
                for r in range(rank):
                    if rng.random() < 0.5:
                        V[j][r] = rng.choice([-1, 1])
            
            for k in range(pn):
                for r in range(rank):
                    if rng.random() < 0.5:
                        W[k][r] = rng.choice([-1, 1])
            
            # Try local repair
            repaired = _local_repair(U, V, W, max_iters=20, deadline=deadline)
            if repaired is not None:
                best = repaired
                best_rank = rank
                break
        
        if best_rank < 42:
            break
    
    if best_rank < 42:
        return best
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
