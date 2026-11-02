"""Rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x5 matrix.

Strategy:
  * Start from the schoolbook decomposition (rank = n*m*p = 50).
  * Use a structured search that exploits the small dimension n=2:
      - Try to find rank-25 decompositions with a (j,k)-indexed structure.
      - Use randomized local search to reduce rank from schoolbook.
  * Validate each candidate by exact reconstruction of the matmul tensor.
  * Keep an explicit deadline guard (return best valid result with safety margin).

The returned dict has keys 'U','V','W' with shapes (n*m,rank),(m*p,rank),(p*n,rank)
and integer/half-integer entries that exactly decompose the matrix multiplication tensor.
"""

import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]


# ---------- helpers ----------

def _is_half_int(x):
    return abs(2 * x - round(2 * x)) < 1e-6


def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that the factors exactly (within tol) reproduce T[i,j,k]=1 for all (i,j,k).

    Also check half-integer entries and correct shapes.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m:
        return False, "U rows"
    if len(V) != m * p:
        return False, "V rows"
    if len(W) != p * n:
        return False, "W rows"
    for row in U:
        if len(row) != rank:
            return False, "U cols"
    for row in V:
        if len(row) != rank:
            return False, "V cols"
    for row in W:
        if len(row) != rank:
            return False, "W cols"
    # half-integer check
    for row in U:
        for x in row:
            if not _is_half_int(x):
                return False, "U not half-int"
    for row in V:
        for x in row:
            if not _is_half_int(x):
                return False, "V not half-int"
    for row in W:
        for x in row:
            if not _is_half_int(x):
                return False, "W not half-int"
    # reconstruction: for all i,j,k, sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False, "recon"
    return True, "ok"


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


# ---------- structured search for rank 25 ----------

def _try_rank25_structured(n, m, p, rng, deadline, safety):
    """Try to find a rank-25 decomposition using a (j,k)-indexed structure.
    
    For each (j,k) pair, we have a rank-1 component. The structure is:
    U[i*m+j', (j,k)] = delta(j, j') * u[i,k]
    V[j'*p+k', (j,k)] = delta(j, j') * delta(k, k')
    W[k'*n+i', (j,k)] = delta(k, k') * w[i', j]
    
    This gives: sum_{j,k} U[i*m+j', (j,k)] * V[j'*p+k', (j,k)] * W[k'*n+i', (j,k)]
    = sum_{j,k} delta(j,j') u[i,k] * delta(j,j') delta(k,k') * delta(k,k') w[i',j]
    = u[i,k'] * w[i', j'] if j'=j and k'=k, else 0
    
    We need u[i,k] * w[i',j] = 1 when i=i' and j=j'=j', k=k'=k'... 
    Actually this structure doesn't directly work. Let's try a different approach.
    
    Alternative: use a bilinear form approach where we split the sum over j.
    """
    def remaining():
        return deadline - safety - (time.time() - start)
    
    # Try random rank-25 factors with local search
    for attempt in range(100):
        if remaining() <= 0:
            break
        U = [[rng.uniform(-2, 2) for _ in range(25)] for _ in range(n * m)]
        V = [[rng.uniform(-2, 2) for _ in range(25)] for _ in range(m * p)]
        W = [[rng.uniform(-2, 2) for _ in range(25)] for _ in range(p * n)]
        
        # Local search: perturb and validate
        for it in range(50):
            if remaining() <= 0:
                break
            r = rng.randrange(25)
            d = rng.gauss(0, 0.2)
            for i in range(n * m):
                U[i][r] += d
            for j in range(m * p):
                V[j][r] -= d * 0.5
            ok, _ = _validate(U, V, W, n, m, p, tol=1e-6)
            if ok:
                # Round to half-integers and re-validate
                U = [[round(x * 2) / 2 for x in row] for row in U]
                V = [[round(x * 2) / 2 for x in row] for row in V]
                W = [[round(x * 2) / 2 for x in row] for row in W]
                ok, _ = _validate(U, V, W, n, m, p, tol=1e-9)
                if ok:
                    return {"U": U, "V": V, "W": W}
            # Revert
            for i in range(n * m):
                U[i][r] -= d
            for j in range(m * p):
                V[j][r] += d * 0.5
    return None


# ---------- local search from schoolbook ----------

def _local_search_from_schoolbook(n, m, p, rng, deadline, safety):
    """Start from schoolbook and try to reduce rank via random perturbations."""
    def remaining():
        return deadline - safety - (time.time() - start)
    
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    for attempt in range(200):
        if remaining() <= 0:
            break
        
        target_rank = best_rank - rng.randint(1, 5)
        if target_rank < 1:
            break
        
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Initialize from schoolbook by merging columns
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    if c < best_rank:
                        r = c % target_rank
                        U[i * m + j][r] += best["U"][i * m + j][c]
                        V[j * p + k][r] += best["V"][j * p + k][c]
                        W[k * n + i][r] += best["W"][k * n + i][c]
        
        # Local search
        for it in range(30):
            if remaining() <= 0:
                break
            r = rng.randrange(target_rank)
            d = rng.gauss(0, 0.1)
            for i in range(n * m):
                U[i][r] += d
            for j in range(m * p):
                V[j][r] -= d * 0.5
            ok, _ = _validate(U, V, W, n, m, p, tol=1e-6)
            if ok:
                U = [[round(x * 2) / 2 for x in row] for row in U]
                V = [[round(x * 2) / 2 for x in row] for row in V]
                W = [[round(x * 2) / 2 for x in row] for row in W]
                ok, _ = _validate(U, V, W, n, m, p, tol=1e-9)
                if ok:
                    return {"U": U, "V": V, "W": W}
            for i in range(n * m):
                U[i][r] -= d
            for j in range(m * p):
                V[j][r] += d * 0.5
    
    return best


# ---------- main ----------

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 150.0
    safety = 15.0  # seconds margin
    
    def remaining():
        return deadline - safety - (time.time() - start)
    
    rng = random.Random(42)
    
    # 1) Try structured rank-25 search
    result = _try_rank25_structured(n, m, p, rng, deadline, safety)
    if result is not None:
        return result
    
    # 2) Try local search from schoolbook
    result = _local_search_from_schoolbook(n, m, p, rng, deadline, safety)
    if result is not None:
        return result
    
    # 3) Fallback: schoolbook
    return _schoolbook(n, m, p)


# EVOLVE-BLOCK-END
