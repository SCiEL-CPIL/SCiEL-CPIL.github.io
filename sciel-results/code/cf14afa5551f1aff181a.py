"""Matrix multiplication 3,4,4 rank-minimal decomposition search."""
import time
import numpy as np
import itertools

DIMENSIONS = [3, 4, 4]

def _make_schoolbook(n, m, p):
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
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != r for row in U):
        return False
    if any(len(row) != r for row in V):
        return False
    if any(len(row) != r for row in W):
        return False
    
    # Verify tensor equation: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 if the (i,j,k) triple corresponds to a valid multiplication term
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k 
    # (the product (AB)[i,k] = sum_j A[i,j] B[j,k])
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _try_rational_rounding(U, V, W, tol=1e-6):
    """Try to round entries to nearby rationals with small denominators."""
    from fractions import Fraction
    r = len(U[0])
    new_U = [[0] * r for _ in range(len(U))]
    new_V = [[0] * r for _ in range(len(V))]
    new_W = [[0] * r for _ in range(len(W))]
    
    def round_frac(x):
        if abs(x) < tol:
            return 0
        # Try fractions with small denominators
        for denom in range(1, 20):
            num = round(x * denom)
            if abs(x - num / denom) < tol:
                if num % denom == 0:
                    return num // denom
                return num / denom
        return x
    
    for i in range(len(U)):
        for c in range(r):
            new_U[i][c] = round_frac(U[i][c])
    for i in range(len(V)):
        for c in range(r):
            new_V[i][c] = round_frac(V[i][c])
    for i in range(len(W)):
        for c in range(r):
            new_W[i][c] = round_frac(W[i][c])
    
    if _validate(new_U, new_V, new_W, DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2]):
        return {"U": new_U, "V": new_V, "W": new_W}
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s
    
    n, m, p = DIMENSIONS
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try some known smaller constructions
    # For 3x4x4, the schoolbook rank is 48.
    # We can try to find lower ranks using random initialization and gradient descent,
    # or by trying structured decompositions.
    
    # Strategy 1: Try random rank-r decompositions with r < 48 and check validity
    # Strategy 2: Use the fact that we can factor out structure
    
    # Actually, for matrix multiplication, we know that:
    # rank(3,4,4) <= rank(3,3,3) * rank(4,4,4) / something... no that's not right.
    
    # Let's try: since n=3, m=4, p=4, we can think of it as:
    # (3x4) * (4x4) = 3x4
    # We can partition the 4x4 into blocks or use block structure.
    
    # A simple improvement: if we can factor the m=4 dimension as 2x2, we might save.
    # But the tensor structure doesn't allow simple factorization.
    
    # Let's try random search for lower rank with integer/half-integer entries.
    import random
    random.seed(42)
    
    # For small ranks, try random sparse patterns
    for target_rank in range(24, best_rank):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
            
        # Try a few random sparse decompositions
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Create random sparse U, V, W with entries from {-1, 0, 1}
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Each column c should have non-zero entries in a structured way
            for c in range(target_rank):
                # Pick random positions
                ui = random.randint(0, n * m - 1)
                vj = random.randint(0, m * p - 1)
                wk = random.randint(0, p * n - 1)
                U[ui][c] = random.choice([-1, 1])
                V[vj][c] = random.choice([-1, 1])
                W[wk][c] = random.choice([-1, 1])
            
            if _validate(U, V, W, n, m, p):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    # Strategy: Try structured decompositions based on known results
    # For matrix multiplication, Strassen-like decompositions can help.
    # But for 3x4x4, let's try to see if we can get below 48.
    
    # Actually, let's try a different approach: use the fact that 
    # we can write the product as a sum of rank-1 terms where each term
    # corresponds to a specific (i,j,k) pattern.
    
    # The key insight: T[i,j,k] = 1 for all i in [0,2], j in [0,3], k in [0,3]
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is equivalent to finding a decomposition of the all-ones tensor
    # in the "multiplied" sense.
    
    # Let's try to use a basis expansion approach.
    # If we set U, V, W to be indicator matrices for subsets, we can 
    # represent the tensor as a sum of products.
    
    # For now, return the best we've found.
    if best_rank < n * m * p:
        return best
    else:
        return best
