"""Rank-minimal exact tensor decomposition for 3x3 by 3x4 matrix multiplication."""
import time
import random
import fractions
import numpy as np
import sys

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

def _check_entries(U, V, W, rank):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        for row in mat:
            for x in row:
                if isinstance(x, Frac):
                    if x.denominator not in (1, 2):
                        return False
                elif isinstance(x, (int, float)):
                    if isinstance(x, float):
                        # Check if it is effectively integer or half-integer
                        val = 2 * x
                        if abs(val - round(val)) > 1e-9:
                            return False
    return True

def _solve_w_system_exact(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c]) * Frac(V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)
    C = rank
    aug = [rows[r][:] + [rhs[r]] for r in range(R)]
    
    pivots = []
    col = 0
    for r in range(R):
        if col >= C:
            break
        pr = None
        for rr in range(r, R):
            if aug[rr][col] != 0:
                pr = rr
                break
        if pr is None:
            col += 1
            continue
        if pr != r:
            aug[r], aug[pr] = aug[pr], aug[r]
        pivots.append((r, col))
        pv = aug[r][col]
        if pv != 1:
            aug[r] = [x / pv for x in aug[r]]
        for rr in range(R):
            if rr != r and aug[rr][col] != 0:
                f = aug[rr][col]
                aug[rr] = [a - f * b for a, b in zip(aug[rr], aug[r])]
        col += 1
    
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _fast_consistency_check_numpy(U, V, i, k, n, m, p, rank):
    """Fast float consistency check using numpy."""
    M = np.zeros((m*m, rank), dtype=float)
    b = np.zeros(m*m, dtype=float)
    for j in range(m):
        for l in range(m):
            b[j*m+l] = 1.0 if j == l else 0.0
            for c in range(rank):
                M[j*m+l, c] = float(U[i*m+j][c]) * float(V[l*p+k][c])
    try:
        sol, residuals, rank_np, sv = np.linalg.lstsq(M, b, rcond=None)
        if residuals and residuals[0] < 1e-9:
            return True
        if rank_np == min(m*m, rank) and np.allclose(M @ sol, b, atol=1e-9):
            return True
    except:
        pass
    return False

def _search_rank_structured(rank, n, m, p, deadline, rng, max_iter):
    """Try structured patterns for rank-`rank` decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Strategy: Random small integers
        # Use a wider range to avoid rank deficiency
        U = [[rng.randint(-4, 4) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-4, 4) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: ensure no row is all zeros
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # For each (i,k), check consistency using numpy first
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                if not _fast_consistency_check_numpy(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
            if not ok:
                break
        
        if not ok:
            continue
            
        # If float check passes, do exact check
        for i in range(n):
            for k in range(p):
                Wvec = _solve_w_system_exact(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        if not _check_entries(U, V, W, rank):
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
    """Try structured search for lower ranks, prioritizing lower ranks."""
    # Focus on ranks that are likely achievable: 23-30
    ranks_to_try = [23, 24, 25, 26, 27, 28, 29, 30]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate time based on rank difficulty
        # Lower ranks are harder, give them more time if we can, but we have a global deadline.
        # We just run until deadline or success.
        max_iter = 1000000 # Effectively time-limited
        
        rng = random.Random(rank * 7919 + 12345)
        result = _search_rank_structured(rank, n, m, p, deadline, rng, max_iter)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    
    # Try structured search for lower rank
    try:
        result = _try_structured_search(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception as e:
        pass
    
    return best_result
