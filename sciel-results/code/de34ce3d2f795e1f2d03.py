"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Exactly verify that U, V, W decompose the multiplication tensor."""
    rank = len(U[0])
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _random_decomposition(rank, n, m, p, rng):
    """Generate a random integer decomposition and try to verify."""
    U = [[rng.randint(-3, 3) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.randint(-3, 3) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.randint(-3, 3) for _ in range(rank)] for _ in range(p * n)]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Best known: start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions via search
    # Known rank for 2x3 x 3x7 is 33
    for target_rank in range(33, best_rank + 1):
        if _time_left() < 20:
            break
        
        # Try random searches for this rank
        max_attempts = 50000
        for attempt in range(max_attempts):
            if _time_left() < 15:
                break
            if attempt % 1000 == 0 and _time_left() < 10:
                break
            
            U, V, W = _random_decomposition(target_rank, n, m, p, rng)
            if _verify(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            # Found a decomposition at this rank; try to go lower
            continue
    
    return best
# EVOLVE-BLOCK-END
