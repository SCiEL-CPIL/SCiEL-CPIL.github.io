"""Rewritten program: search for a lower-rank exact tensor decomposition of the 4x4 * 4x5 matrix multiplication tensor."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _construct_schoolbook():
    """Return the trivial rank-80 decomposition."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta_{i,k} for all i,j,k."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We want T[i,j,k] = 1 if i==k else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1 if i == k else 0
                if s != expected:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy with integer arithmetic (exact for small ints)."""
    rank = len(U[0])
    Ua = np.array(U, dtype=object)  # exact integer objects
    Va = np.array(V, dtype=object)
    Wa = np.array(W, dtype=object)
    # Reshape: Ua[i*m+j, c], Va[j*p+k, c], Wa[k*n+i, c]
    # For each c, the contribution to T[i,j,k] is Ua[i*m+j,c]*Va[j*p+k,c]*Wa[k*n+i,c]
    # We need T[i,j,k] = sum_c ... = delta_{i,k}
    # Build T as a 3D array
    T = np.zeros((n, m, p), dtype=object)
    for c in range(rank):
        uc = Ua[:, c]  # shape (n*m,)
        vc = Va[:, c]  # shape (m*p,)
        wc = Wa[:, c]  # shape (p*n,)
        # uc[i*m+j], vc[j*p+k], wc[k*n+i]
        # For fixed i,j,k: uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
        # Sum over i,j,k
        for i in range(n):
            for j in range(m):
                uval = uc[i * m + j]
                if uval == 0:
                    continue
                for k in range(p):
                    vval = vc[j * p + k]
                    if vval == 0:
                        continue
                    wval = wc[k * n + i]
                    if wval == 0:
                        continue
                    T[i, j, k] += uval * vval * wval
    # Check T[i,j,k] == 1 if i==k else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if i == k else 0
                if T[i, j, k] != expected:
                    return False
    return True

def _try_rank_random(rank, n, m, p, max_iters, seed=0, deadline=DEADLINE):
    """Try to find a valid decomposition of given rank using random search."""
    rng = random.Random(seed)
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        # Generate random integer matrices with entries in {-1, 0, 1}
        U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p * n)]
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None

def _try_rank_structured(rank, n, m, p, max_iters, seed=0, deadline=DEADLINE):
    """Try structured decompositions: fix some entries to exploit symmetry."""
    rng = random.Random(seed)
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        # Strategy: use a subset of the schoolbook terms plus random perturbations
        # Start with schoolbook and try to merge terms
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Assign random values to first rank entries
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = rng.choice([-1, 0, 1])
            for j in range(m):
                for k in range(p):
                    V[j * p + k][c] = rng.choice([-1, 0, 1])
            for k in range(p):
                for i in range(n):
                    W[k * n + i][c] = rng.choice([-1, 0, 1])
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the known schoolbook solution (rank 80)
    best = _construct_schoolbook()
    best_rank = n * m * p  # 80
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Also rank >= n + m + p - 2 = 11 (weak)
    # For 4x4 * 4x5, the Strassen-like bounds suggest rank around 25-35
    
    # Try ranks from 20 upward
    for rank in range(20, best_rank):
        if time.time() > DEADLINE - 10:
            break
        
        # Try random search
        max_iters = 5000
        result = _try_rank_random(rank, n, m, p, max_iters, seed=rank, deadline=DEADLINE - 5)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        # Try structured search
        if time.time() > DEADLINE - 10:
            break
        result = _try_rank_structured(rank, n, m, p, max_iters // 2, seed=rank + 1000, deadline=DEADLINE - 5)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
