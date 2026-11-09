"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time as _time
import random as _random
import numpy as _np

def construct():
    _start = _time.time()
    _deadline = _start + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    _sb = _schoolbook(n, m, p)
    _best = {"U": _sb[0], "V": _sb[1], "W": _sb[2]}
    _best_rank = n * m * p
    
    # Try structured constructions first for better initial bounds
    _strassen = _strassen_5x5x5(n, m, p)
    if _strassen is not None:
        _best = _strassen
        _best_rank = _strassen_rank(_strassen)
    
    # Try to find lower-rank decompositions using known constructions and search
    # Known lower bound: R(5,5,5) >= 49 (from Coppersmith-Winograd type bounds, but actual LB is lower)
    # Try ranks from a reasonable lower bound upward
    for _rank in range(49, _best_rank):
        if _time.time() > _deadline:
            break
        _res = _search_rank(n, m, p, _rank, _deadline)
        if _res is not None:
            _best = _res
            _best_rank = _rank
            break  # Found smallest rank in this range
    
    return _best

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

def _strassen_rank(result):
    """Get the rank from a result dict."""
    if result is None:
        return float('inf')
    U = result['U']
    if isinstance(U, list):
        return len(U[0]) if U else 0
    else:
        return U.shape[1]

def _strassen_5x5x5(n, m, p):
    """
    Construct a 5x5x5 decomposition using block structure.
    Split 5 = 2 + 3, use R(2,2,2)=7 and R(3,3,3) constructions.
    """
    # For 5x5x5, we can use the fact that:
    # [A1 A2; A3 A4] * [B1 B2; B3 B4]
    # where A1,B1 are 3x3, A2,B2 are 3x2, A3,B3 are 2x3, A4,B4 are 2x2
    # 
    # This is complex. Let's try a simpler approach:
    # Use a known construction for R(5,5,5) if available, 
    # or just return None to let the search handle it.
    
    # Actually, let's try a direct construction using the "sum of products" idea
    # with a structured pattern.
    
    # For now, return None
    return None

def _search_rank(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition. Returns dict or None."""
    # Use a structured search: exploit that the tensor is 1 for all (i,j,k)
    
    # Strategy 1: Random restarts with small integer entries
    for _attempt in range(100):
        if _time.time() > deadline:
            return None
        _res = _try_random(n, m, p, rank)
        if _res is not None:
            return _res
    
    # Strategy 2: Structured initialization
    for _attempt in range(50):
        if _time.time() > deadline:
            return None
        _res = _try_structured(n, m, p, rank)
        if _res is not None:
            return _res
    
    return None

def _try_random(n, m, p, rank):
    """Try a random initialization and check if it works."""
    # Initialize with small integers
    U = [[_random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
    V = [[_random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
    W = [[_random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
    
    # Check constraint: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return None
    return {"U": U, "V": V, "W": W}

def _try_structured(n, m, p, rank):
    """Try a structured initialization that exploits tensor symmetry."""
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Try a construction based on the "all-ones" structure
    # Idea: use rank components that each contribute to a subset of (i,j,k)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pattern: use a modular arithmetic structure
    mod = max(1, rank // 5)
    
    for r in range(rank):
        # Each rank component r handles indices where (i + 2*j + 3*k) % mod == r % mod
        for i in range(n):
            for j in range(m):
                if (i + 2 * j) % mod == r % mod:
                    U[i * m + j][r] = 1
        for j in range(m):
            for k in range(p):
                if (j + 2 * k) % mod == r % mod:
                    V[j * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                if (k + 2 * i) % mod == r % mod:
                    W[k * n + i][r] = 1
    
    # Check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return None
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
