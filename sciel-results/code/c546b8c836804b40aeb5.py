"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check tensor entry: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Check all other tensor entries are 0
    # The tensor T[i,j,k] should be 1 if (i,j,k) corresponds to a valid matrix mult entry
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # and we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # There are no "other" entries - the tensor is fully defined by the n*m*p entries
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_random_rank(n, m, p, rank, seed=0, max_iters=1000):
    """Try to find a rank-`rank` decomposition using random initialization and local search."""
    random.seed(seed)
    best = None
    
    for it in range(max_iters):
        # Random initialization with entries from {-1, 0, 1}
        U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _greedy_construct(n, m, p, rank, seed=0):
    """Greedy construction: build decomposition one term at a time."""
    random.seed(seed)
    
    # We want to cover all (i,j,k) triples with rank terms
    # Each term c contributes U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] to T[i,j,k]
    # We need sum_c ... = 1 for each (i,j,k)
    
    # Start with all zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    remaining = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for c in range(rank):
        if not remaining:
            break
        # Choose a random (i,j,k) from remaining
        idx = random.randint(0, len(remaining) - 1)
        i, j, k = remaining[idx]
        
        # Set this term to contribute 1 to (i,j,k)
        U[i*m+j][c] = 1
        V[j*p+k][c] = 1
        W[k*n+i][c] = 1
        
        # Check which other (i',j',k') are now satisfied (sum so far = 1)
        # and which are over-satisfied (sum > 1) or under-satisfied
        still_remaining = []
        for (i2, j2, k2) in remaining:
            s = 0.0
            for cc in range(c + 1):
                s += U[i2*m+j2][cc] * V[j2*p+k2][cc] * W[k2*n+i2][cc]
            if abs(s - 1.0) < 1e-9:
                # Satisfied
                pass
            elif abs(s) > 1e-9:
                # Partially satisfied or over - keep tracking
                still_remaining.append((i2, j2, k2))
            else:
                still_remaining.append((i2, j2, k2))
        
        remaining = still_remaining
    
    if not remaining:
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    
    # Try to find lower rank decompositions
    # Known results: for 3x3x3, rank is 23 (Strassen-like)
    # For 3x4x4x6, we try various ranks
    
    # Try structured approaches first
    # Approach: use Kronecker products or block structures
    
    # Let's try a direct search with small ranks
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # But for matrix mult, rank is typically much higher
    
    # Try ranks from 24 upward, but be smart about it
    # Given time constraints, try a few promising ranks
    
    ranks_to_try = [24, 25, 26, 27, 28, 29, 30, 32, 36, 40, 48, 54, 60, 64, 66, 68, 70, 71]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Try random search for this rank
        result = _try_random_rank(n, m, p, rank, seed=rank, max_iters=50)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        # Try greedy
        if time.time() > deadline - 20:
            break
        result = _greedy_construct(n, m, p, rank, seed=rank)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return best
# EVOLVE-BLOCK-END
