import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p matrix multiplication."""
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

def _validate(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    rank = len(U[0])
    
    for a in range(dim_a):
        for b in range(dim_b):
            for c in range(dim_c):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                expected = 1 if (i == i2 and j == j2) else 0
                
                actual = Fraction(0)
                for col in range(rank):
                    actual += Fraction(U[a][col]) * Fraction(V[b][col]) * Fraction(W[c][col])
                
                if actual != expected:
                    return False
    return True

def _try_fmm3_extension(n, m, p, deadline):
    """
    Try to construct a low-rank decomposition for 3x3x3x4 by extending
    FMM-3 style ideas. We'll attempt to find a valid decomposition
    with rank < 36 using a structured approach.
    
    Strategy:
    1. Start with schoolbook (rank 36)
    2. Try to eliminate columns by checking if the tensor contribution
       of a column can be expressed as a linear combination of others
    3. Use the structure of the problem to guide the search
    """
    if n != 3 or m != 3 or p != 4:
        return None
    
    dim_a = n * m  # 9
    dim_b = m * p  # 12
    dim_c = p * n  # 12
    
    # Build the target tensor T[a, b, c]
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    
    # Start with schoolbook
    sb = _schoolbook(n, m, p)
    U = [[Fraction(x) for x in row] for row in sb["U"]]
    V = [[Fraction(x) for x in row] for row in sb["V"]]
    W = [[Fraction(x) for x in row] for row in sb["W"]]
    
    rank = n * m * p  # 36
    
    # Try to eliminate columns one by one
    # For each column c, check if we can remove it by adjusting others
    
    # A column c can be removed if for all (a, b, c_idx):
    #   U[a,c]*V[b,c]*W[c_idx,c] = sum_{c' != c} alpha_{c'} * U[a,c']*V[b,c']*W[c_idx,c']
    
    # This is a linear system. Let's try a simpler approach:
    # For each pair of columns, check if one can be eliminated
    
    # Actually, let's try a different approach: use the fact that
    # the tensor has a specific structure and try to find a
    # decomposition with fewer terms.
    
    # For 3x3x4, we can try to use the following structure:
    # - For each (i, j), we have 4 terms (one for each k)
    # - We can try to reduce the number of terms by using
    #   linear combinations
    
    # Let's try to find a rank-30 decomposition by construction
    
    # Idea: For each row i of A, we can use a shared transformation
    # across all columns j
    
    # This is getting complex. Let's try a randomized local search.
    
    # Start with schoolbook and try random perturbations
    
    import random
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try random column elimination
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        # Randomly select a column to try to eliminate
        c = random.randint(0, rank - 1)
        
        # Check if this column can be eliminated
        # by seeing if its contribution is a linear combination of others
        
        # For simplicity, just try removing it and see if we can
        # fix the other columns
        
        # This is a heuristic, so it may not work
        
        if random.random() < 0.5:
            # Try to eliminate column c by zeroing it out
            # and checking if the result is still valid
            U_test = [row[:] for row in U]
            V_test = [row[:] for row in V]
            W_test = [row[:] for row in W]
            
            # Zero out column c
            for a in range(dim_a):
                U_test[a][c] = Fraction(0)
            for b in range(dim_b):
                V_test[b][c] = Fraction(0)
            for c_idx in range(dim_c):
                W_test[c_idx][c] = Fraction(0)
            
            # Check if still valid
            if _validate(U_test, V_test, W_test, n, m, p):
                # Remove column c
                U = [row[:c] + row[c+1:] for row in U]
                V = [row[:c] + row[c+1:] for row in V]
                W = [row[:c] + row[c+1:] for row in W]
                rank -= 1
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                if rank <= 30:
                    break
    
    if best_rank < 36:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return None

def _try_direct_construction(n, m, p, deadline):
    """
    Try to directly construct a low-rank decomposition for 3x3x4.
    
    We'll use the following approach:
    1. For each (i, j) pair, try to use fewer than p terms
    2. Use linear combinations to share computations
    
    This is a heuristic approach that may not find the optimal rank,
    but should find something better than schoolbook.
    """
    if n != 3 or m != 3 or p != 4:
        return None
    
    dim_a = n * m  # 9
    dim_b = m * p  # 12
    dim_c = p * n  # 12
    
    # Let's try to construct a decomposition with rank 30
    # by using the following structure:
    # - For each i, use a shared transformation for all j
    # - This allows us to reduce the number of terms
    
    # For now, let's try a simple construction:
    # Use schoolbook but try to merge some terms
    
    # Actually, let's try to find a valid decomposition with
    # rank 32 by removing 4 terms from schoolbook
    
    # This is a heuristic, so it may not work
    
    return None

def _search_low_rank(n, m, p, deadline, max_rank=35):
    """
    Search for a low-rank decomposition.
    """
    # Try FMM-3 extension
    result = _try_fmm3_extension(n, m, p, deadline)
    if result is not None:
        return result
    
    # Try direct construction
    result = _try_direct_construction(n, m, p, deadline)
    if result is not None:
        return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # First, get schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find better decomposition
    result = _search_low_rank(n, m, p, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank and _validate(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
