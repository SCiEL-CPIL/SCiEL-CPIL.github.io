import time
import random
import fractions
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
Frac = Fraction

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

def _try_rank_construction(rank, n, m, p, deadline, rng):
    """Try to find a valid decomposition at given rank using structured search."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    max_iter = 5000
    it = 0
    
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Generate random U and V with small integers
        # Strategy: dense random with {-1, 0, 1}
        U = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: no all-zero rows
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # Build coefficient matrices for solving W
        # For each (i, k), we have m*m equations in rank unknowns
        # A[i,k][j*m+l, c] = U[i*m+j, c] * V[l*p+k, c]
        # b[i,k][j*m+l] = 1 if j==l else 0
        
        # First, check consistency using numpy
        A_all = []
        b_all = []
        for i in range(n):
            for k in range(p):
                A = np.zeros((m*m, rank))
                b = np.zeros(m*m)
                for j in range(m):
                    for l in range(m):
                        for c in range(rank):
                            A[j*m + l, c] = float(U[i*m + j][c]) * float(V[l*p + k][c])
                        b[j*m + j] = 1.0
                A_all.append(A)
                b_all.append(b)
        
        # Check consistency for each system
        feasible = True
        for idx in range(n * p):
            A = A_all[idx]
            b = b_all[idx]
            try:
                rank_A = np.linalg.matrix_rank(A, tol=1e-10)
                Ab = np.column_stack([A, b])
                rank_Ab = np.linalg.matrix_rank(Ab, tol=1e-10)
                if rank_A != rank_Ab:
                    feasible = False
                    break
            except:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Solve for W using exact arithmetic
        W = [[0] * rank for _ in range(pn)]
        ok = True
        for idx in range(n * p):
            i = idx // p
            k = idx % p
            A = A_all[idx]
            b = b_all[idx]
            
            # Convert to fractions and solve
            A_frac = [[Frac(A[r][c]) for c in range(rank)] for r in range(m*m)]
            b_frac = [Frac(b[r]) for r in range(m*m)]
            
            # Gaussian elimination
            R = m * m
            C = rank
            aug = [A_frac[r][:] + [b_frac[r]] for r in range(R)]
            
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
            
            # Check for inconsistency
            for rr in range(R):
                coeff_zero = all(aug[rr][c] == 0 for c in range(C))
                if coeff_zero and aug[rr][C] != 0:
                    ok = False
                    break
            if not ok:
                break
            
            Wvec = [Frac(0)] * C
            for (r, c) in pivots:
                Wvec[c] = aug[r][C]
            
            W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
        
        if not ok:
            continue
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from low to high
    for rank in range(18, best_rank):
        if time.time() > deadline:
            break
        
        rng = random.Random(rank * 1000 + 42)
        result = _try_rank_construction(rank, n, m, p, deadline, rng)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best_result
