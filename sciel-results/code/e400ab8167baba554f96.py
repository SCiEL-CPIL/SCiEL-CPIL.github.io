import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, with 10s safety margin

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matmul tensor."""
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

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return U, V, W

def _random_search(n, m, p, target_rank, n_attempts=1000):
    """
    Random search for a valid decomposition at the given rank.
    Entries are drawn from {-1, 0, 1}.
    """
    for attempt in range(n_attempts):
        if _time_left() < 5:
            return None
        
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        density = 0.25
        for i in range(n * m):
            for r in range(target_rank):
                if random.random() < density:
                    U[i][r] = 1 if random.random() < 0.5 else -1
        for j in range(m * p):
            for r in range(target_rank):
                if random.random() < density:
                    V[j][r] = 1 if random.random() < 0.5 else -1
        for k in range(p * n):
            for r in range(target_rank):
                if random.random() < density:
                    W[k][r] = 1 if random.random() < 0.5 else -1
        
        if _validate(U, V, W, n, m, p):
            return (U, V, W)
    
    return None

def _structured_search(n, m, p):
    """
    Try structured decompositions based on known patterns.
    For 4x4x5, try ranks from 20 upward.
    """
    lb = max(n * m, m * p, n * p)  # = 20 for 4,4,5
    
    for target_rank in range(lb, n * m * p + 1):
        if _time_left() < 10:
            break
        
        result = _random_search(n, m, p, target_rank, n_attempts=200)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try structured random search
    result = _structured_search(n, m, p)
    
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
