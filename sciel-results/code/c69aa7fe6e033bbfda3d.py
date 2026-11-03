"""Optimized tensor decomposition for 3x3 * 3x6 matrix multiplication."""
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


def _random_search(n, m, p, rank, max_iter=1000, seed=42):
    """
    Try to find a rank-`rank` decomposition using random search with local refinement.
    """
    deadline = time.time() + _TIME_LIMIT - 15
    rng = random.Random(seed)
    
    # Initialize with small random values
    U = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(n * m)]
    V = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(m * p)]
    W = [[rng.uniform(-1.0, 1.0) for _ in range(rank)] for _ in range(p * n)]
    
    best_err = _compute_error(U, V, W, n, m, p)
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random entry to perturb
        mat_idx = rng.randint(0, 2)
        if mat_idx == 0:
            row = rng.randint(0, n * m - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0])
            old_val = U[row][col]
            U[row][col] += delta
        elif mat_idx == 1:
            row = rng.randint(0, m * p - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0])
            old_val = V[row][col]
            V[row][col] += delta
        else:
            row = rng.randint(0, p * n - 1)
            col = rng.randint(0, rank - 1)
            delta = rng.choice([-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0])
            old_val = W[row][col]
            W[row][col] += delta
        
        err = _compute_error(U, V, W, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        else:
            if mat_idx == 0:
                U[row][col] = old_val
            elif mat_idx == 1:
                V[row][col] = old_val
            else:
                W[row][col] = old_val
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-9:
        return {"U": best_U, "V": best_V, "W": best_W}, best_err
    return None, best_err


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT - 5
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    fallback = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from 27 up to schoolbook
    # Lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Try a few ranks
    for rank in range(27, min(best_rank, 55)):
        if time.time() > _DEADLINE - 10:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > _DEADLINE - 5:
                break
            
            result, err = _random_search(n, m, p, rank, max_iter=500, seed=42 + seed)
            if result is not None:
                # Verify
                U, V, W = result["U"], result["V"], result["W"]
                if _verify(U, V, W, n, m, p):
                    return result
    
    return fallback

# EVOLVE-BLOCK-END
