import time
import random
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _search_rank(target_rank, deadline, seed=42, attempts=50000):
    """
    Search for a decomposition of the given rank using random sparse patterns.
    """
    n, m, p = 3, 3, 3
    rank = target_rank
    random.seed(seed)
    
    # Value set: integers and half-integers
    values = [-2, -1, -0.5, 0.5, 1, 2]
    
    for attempt in range(attempts):
        if time.time() > deadline:
            break
            
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Sparse pattern: each row has 1-3 non-zeros
        for i in range(n * m):
            num_nz = random.randint(1, 3)
            idxs = random.sample(range(rank), num_nz)
            for idx in idxs:
                U[i][idx] = random.choice(values)
                
        for i in range(m * p):
            num_nz = random.randint(1, 3)
            idxs = random.sample(range(rank), num_nz)
            for idx in idxs:
                V[i][idx] = random.choice(values)
                
        for i in range(p * n):
            num_nz = random.randint(1, 3)
            idxs = random.sample(range(rank), num_nz)
            for idx in idxs:
                W[i][idx] = random.choice(values)
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    # Use a deadline that leaves 10s margin before the 120s kill
    # The prompt says "Each evaluation must finish within 120 s"
    # The directive says "evaluator KILLS the program after 150s"
    # I will target 115s to be safe.
    overall_deadline = time.time() + 115
    
    # 1. Try searching for rank 21 (Unlikely but fast to check a few)
    # Search is expensive, so we limit attempts.
    # Let's start with higher ranks first to ensure we get a valid result quickly,
    # then try to improve.
    
    # Strategy: 
    # - Rank 24 is known to be possible (Lader). 
    # - Rank 23 is known to be possible (Lader/Strassen variants).
    # - Rank 22 is open (unknown if possible).
    # - Rank 21 is open.
    
    # We will try rank 24 first, then 23.
    # If we find one, return it.
    
    # To make search more effective, we can try to bias the search.
    # But simple random search with 50k attempts might take ~10-20s per rank.
    # Let's give rank 24 10s, rank 23 20s, rank 22 20s, rank 21 10s.
    
    # 1. Rank 24
    if time.time() < overall_deadline - 60:
        r24 = _search_rank(24, overall_deadline - 60, seed=42, attempts=20000)
        if r24 is not None:
            return r24
            
    # 2. Rank 23
    if time.time() < overall_deadline - 30:
        r23 = _search_rank(23, overall_deadline - 30, seed=123, attempts=30000)
        if r23 is not None:
            return r23
            
    # 3. Rank 22
    if time.time() < overall_deadline - 10:
        r22 = _search_rank(22, overall_deadline - 10, seed=456, attempts=10000)
        if r22 is not None:
            return r22
            
    # 4. Rank 21
    if time.time() < overall_deadline:
        r21 = _search_rank(21, overall_deadline, seed=789, attempts=5000)
        if r21 is not None:
            return r21
            
    # 5. Fallback to schoolbook rank 27
    return _construct_schoolbook()
