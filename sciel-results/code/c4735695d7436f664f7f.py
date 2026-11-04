"""Rank-minimal exact tensor decomposition search for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np

DIMENSIONS = [3, 3, 4]
Frac = fractions.Fraction

def _validate(U, V, W, n, m, p, rank):
    """Exact check that the tensor decomposition holds."""
    Uf = [[Frac(U[i][c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[Frac(V[j][c]) for c in range(rank)] for j in range(m * p)]
    Wf = [[Frac(W[k][c]) for c in range(rank)] for k in range(p * n)]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for l in range(m):
                    s = Frac(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[l * p + k][c] * Wf[k * n + i][c]
                    expected = Frac(1) if j == l else Frac(0)
                    if s != expected:
                        return False
    return True

def _check_half_integer(W, rank):
    """Check all entries are integer or half-integer."""
    for row in W:
        for x in row:
            if isinstance(x, Frac):
                if x.denominator not in (1, 2):
                    return False
            elif isinstance(x, (int, float)):
                if isinstance(x, float) and x != int(x) and 2*x != int(2*x):
                    return False
    return True

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    range_val = 3

    while time.time() < deadline and it < max_iter:
        it += 1
        # Random integer U,V
        U = [[rng.randint(-range_val, range_val) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-range_val, range_val) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: ensure no row of U or V is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # For each (i,k), check consistency and solve for W
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                M = np.zeros((m * m, rank), dtype=float)
                b = np.zeros(m * m, dtype=float)
                for j in range(m):
                    for l in range(m):
                        b[j * m + l] = 1.0 if j == l else 0.0
                        for c in range(rank):
                            M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
                
                try:
                    W_float, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
                    
                    if not np.allclose(M @ W_float, b, atol=1e-9):
                        ok = False
                        break
                        
                    W_half_int = np.round(W_float * 2) / 2
                    if not np.allclose(W_float, W_half_int, atol=1e-9):
                        ok = False
                        break
                        
                    Wvec = []
                    for x in W_half_int:
                        val = int(round(2 * x))
                        if val % 2 == 0:
                            Wvec.append(Frac(val // 2))
                        else:
                            Wvec.append(Frac(val, 2))
                    
                    for j in range(m):
                        for l in range(m):
                            s = Frac(0)
                            for c in range(rank):
                                s += Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) * Wvec[c]
                            expected = Frac(1) if j == l else Frac(0)
                            if s != expected:
                                ok = False
                                break
                        if not ok:
                            break
                            
                    if not ok:
                        break
                    
                    W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
                except Exception:
                    ok = False
                    break
            
            if not ok:
                break
        
        if not ok:
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
            
    return None

def _schoolbook(n, m, p):
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

def _try_structured_search(n, m, p, deadline):
    """Try structured patterns that might yield lower rank."""
    # Try using Kronecker-like structures
    for rank in range(27, 36):
        rng = random.Random(rank * 1000 + 42)
        result = _search_rank(rank, n, m, p, deadline, rng, max_iter=5000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_result = _schoolbook(n, m, p)
    
    target_ranks = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(rank * 12345 + 678)
        start_rank_time = time.time()
        # Give more time per rank since the search is faster now
        rank_deadline = min(deadline - 5, start_rank_time + 12)
        
        try:
            result = _search_rank(rank, n, m, p, rank_deadline, rng, max_iter=1000000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                if rank < 30:
                    break
        except Exception:
            pass

    return best_result
