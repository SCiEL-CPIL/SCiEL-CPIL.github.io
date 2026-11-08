"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # seconds, with 10s safety margin

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                # For matrix mult tensor, T[i,j,k] should be 1
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _rank_of(U, V, W):
    """Return the rank (number of columns)."""
    if not U or not U[0]:
        return 0
    return len(U[0])

def _try_random_decomposition(n, m, p, target_rank, rng, max_iters=2000):
    """Try to find a decomposition of given rank using random initialization."""
    nm, mp, pn = n*m, m*p, p*n
    
    for _ in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return None
        
        # Random integer/half-integer entries in {-1, 0, 0.5, 1}
        vals = [-1, -0.5, 0, 0.5, 1]
        U = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(nm)]
        V = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(mp)]
        W = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(pn)]
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_structured_decomposition(n, m, p, target_rank):
    """Try structured decompositions based on known patterns."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Try: for each (i,j), assign a subset of columns
    # The idea: U[i*m+j, c] is nonzero only for specific c
    # This is like a sparse assignment
    
    # Approach: Use the fact that for matrix mult, we need
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # Try rank = max(n*m, m*p, n*p) with structured patterns
    # For 3x3 x 3x6: nm=9, mp=18, pn=18
    
    # Pattern 1: U depends only on i, V on j,k, W on k,i
    # U[i*m+j, c] = u_i[c] (same for all j)
    # Then sum_c u_i[c] * V[j,k,c] * W[k,i,c] = 1 for all j
    # This requires V[j,k,c]*W[k,i,c] to be the same for all j, which is restrictive
    
    # Pattern 2: Use Kronecker-like structure
    # Split into blocks
    
    # Try exhaustive small-rank search for very small ranks
    if target_rank <= 12:
        # Try all combinations of {-1, 0, 1} for small rank
        # This is infeasible for large dimensions, so only try if rank is small
        pass
    
    return None

def _greedy_improve(U, V, W, n, m, p, max_rounds=100):
    """Try to reduce rank by removing columns that don't contribute."""
    rank = _rank_of(U, V, W)
    current_rank = rank
    
    for _ in range(max_rounds):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        if current_rank <= max(n*m, m*p, p*n):
            break
        
        # Try removing each column and check if still valid
        removed = False
        for c in range(current_rank):
            # Temporarily remove column c
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            
            if _validate(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    return {"U": U, "V": V, "W": W}

def _try_rank_search(n, m, p, start_rank, end_rank, rng):
    """Search for the minimum rank decomposition."""
    best_result = None
    best_rank = float('inf')
    
    for rank in range(start_rank, end_rank + 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try random decomposition
        result = _try_random_decomposition(n, m, p, rank, rng, max_iters=500)
        if result is not None:
            best_rank = rank
            best_result = result
            break  # Found minimum rank, stop
    
    return best_result, best_rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    lower_bound = max(nm, mp, pn)
    
    # Start with schoolbook as fallback
    fallback = _make_schoolbook(n, m, p)
    best_result = fallback
    best_rank = schoolbook_rank
    
    rng = random.Random(42)
    
    # Try to find lower rank decompositions
    # Search from lower bound upward
    search_start = lower_bound
    search_end = min(schoolbook_rank, lower_bound + 20)  # Limit search space
    
    for rank in range(search_start, search_end + 1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try random search for this rank
        result = _try_random_decomposition(n, m, p, rank, rng, max_iters=1000)
        if result is not None:
            best_rank = rank
            best_result = result
            break
    
    # Try to improve with greedy column removal
    if best_rank < schoolbook_rank:
        improved = _greedy_improve(
            [row[:] for row in best_result["U"]],
            [row[:] for row in best_result["V"]],
            [row[:] for row in best_result["W"]],
            n, m, p
        )
        imp_rank = _rank_of(improved["U"], improved["V"], improved["W"])
        if imp_rank < best_rank:
            best_rank = imp_rank
            best_result = improved
    
    return best_result

def _make_schoolbook(n, m, p):
    """Create the schoolbook decomposition."""
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
# EVOLVE-BLOCK-END
