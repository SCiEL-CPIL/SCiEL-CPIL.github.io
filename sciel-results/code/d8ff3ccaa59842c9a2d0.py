import time
import random
import numpy as np

DIMENSIONS = [2, 2, 6]

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Reconstruct tensor: for each (i,k,j), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == delta(i,j valid) * delta(j,k valid)
    # Actually, the tensor T[i,k,j] should be 1 if the (i,j) of A and (j,k) of B multiply to C[i,k]
    # T[i,k,j] = 1 for all valid i,j,k (since every product contributes)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                expected = 1
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _search_random(n, m, p, max_rank, deadline, rng):
    """Try to find a decomposition of rank <= max_rank using random initialization."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for r in range(2, max_rank + 1):
        attempts = 0
        while time.time() < deadline - 5:
            attempts += 1
            # Random initialization with small integers
            U = [[rng.choice([-1, 0, 1, 2, -2]) for _ in range(r)] for _ in range(nm)]
            V = [[rng.choice([-1, 0, 1, 2, -2]) for _ in range(r)] for _ in range(mp)]
            W = [[rng.choice([-1, 0, 1, 2, -2]) for _ in range(r)] for _ in range(pn)]
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _search_structural(n, m, p, deadline, rng):
    """Structured search: use the fact that each term can cover multiple (i,k) pairs."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Try rank 8-12 with structured patterns
    for r in range(8, min(16, n*m*p + 1)):
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            # Initialize with patterns
            U = [[0] * r for _ in range(nm)]
            V = [[0] * r for _ in range(mp)]
            W = [[0] * r for _ in range(pn)]
            
            for c in range(r):
                # Pick a random j for this column
                j = rng.randint(0, m - 1)
                # U: pick which i values this term covers
                u_val = rng.choice([-1, 1, 2, -2])
                for i in range(n):
                    if rng.random() < 0.7:
                        U[i * m + j][c] = u_val
                # V: pick which k values
                v_val = rng.choice([-1, 1, 2, -2])
                for k in range(p):
                    if rng.random() < 0.5:
                        V[j * p + k][c] = v_val
                # W: pick which (k,i) pairs
                w_val = rng.choice([-1, 1, 2, -2])
                for k in range(p):
                    for i in range(n):
                        if rng.random() < 0.4:
                            W[k * n + i][c] = w_val
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s kill
    
    # First, always have the schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    rng = random.Random(42)
    
    # Try structured search first
    if time.time() < deadline - 10:
        result = _search_structural(n, m, p, deadline, rng)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    # Try random search
    if time.time() < deadline - 10:
        result = _search_random(n, m, p, best_rank - 1, deadline, rng)
        if result is not None:
            r = len(result["U"][0])
            if r < best_rank:
                best = result
                best_rank = r
    
    return best
