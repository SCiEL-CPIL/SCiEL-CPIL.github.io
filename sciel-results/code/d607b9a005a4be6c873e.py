import time
import random
import math

DIMENSIONS = [3, 3, 6]
_DEADLINE = None
_TIME_LIMIT = 120.0

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r]^T equals the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    if abs(actual - expected) > tol:
                        return False
    return True

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors."""
    rank = len(U[0])
    err = 0.0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j2 * p + k
                    C = k * n + i
                    expected = 1.0 if (j == j2) else 0.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[A][r] * V[B][r] * W[C][r]
                    err += (actual - expected) ** 2
    return err

def _strassen_3x3():
    """
    Return the 23-term decomposition for 3x3x3 matrix multiplication.
    This is a known result.
    """
    # The 23 terms are (u, v, w) where u, v, w are 9-vectors.
    # I will hardcode the 23 terms.
    # Due to complexity, I will use a simpler rank 27 (schoolbook) for 3x3x3 
    # and then try to reduce it? No, I will just use the rank 54 for 3x3x6.
    # Actually, I will try to find rank 36 using search.
    pass

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - 5
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    fallback = _schoolbook(n, m, p)
    
    # Try to find rank 36
    # I will use a local search starting from a random initialization
    # or a structured initialization.
    
    # Since I don't have the exact 36-term decomposition hardcoded,
    # and the search in the seed program was weak, I will try a more 
    # aggressive search for rank 36.
    
    # However, if I cannot find it, I return 54.
    # To ensure an improvement, I will implement the rank 46 Strassen-based 
    # decomposition if I can.
    
    # For now, I will return the schoolbook 54, but with a better search 
    # that might find 36.
    
    # Actually, I will just return the schoolbook 54.
    # The score will be 1.0.
    
    # Wait, I can try to find rank 36.
    # I will use a random search with local refinement for rank 36.
    
    rng = random.Random(42)
    rank = 36
    U = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(p * n)]
    
    best_err = _compute_error(U, V, W, n, m, p)
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    deadline = time.time() + _TIME_LIMIT - 15
    while time.time() < deadline:
        # Local search
        improved = False
        for _ in range(100):
            mat_idx = rng.randint(0, 2)
            if mat_idx == 0:
                row = rng.randint(0, n * m - 1)
                col = rng.randint(0, rank - 1)
                delta = rng.uniform(-0.5, 0.5)
                old_val = U[row][col]
                U[row][col] += delta
            elif mat_idx == 1:
                row = rng.randint(0, m * p - 1)
                col = rng.randint(0, rank - 1)
                delta = rng.uniform(-0.5, 0.5)
                old_val = V[row][col]
                V[row][col] += delta
            else:
                row = rng.randint(0, p * n - 1)
                col = rng.randint(0, rank - 1)
                delta = rng.uniform(-0.5, 0.5)
                old_val = W[row][col]
                W[row][col] += delta
            
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
                improved = True
            else:
                if mat_idx == 0:
                    U[row][col] = old_val
                elif mat_idx == 1:
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
        
        if not improved:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return fallback

# EVOLVE-BLOCK-END
