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

def _search_rank_integer(rank, n, m, p, deadline, rng, max_iter):
    """
    Search for integer U', V', W' such that sum U' V' W' = 8 * delta.
    Then return U=U'/2, V=V'/2, W=W'/2.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Generate random integer U', V' in {-2, -1, 0, 1, 2}
        # We want them to be somewhat dense but not too large
        U_prime = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(nm)]
        V_prime = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: no all-zero rows
        if any(all(x == 0 for x in row) for row in U_prime):
            continue
        if any(all(x == 0 for x in row) for row in V_prime):
            continue
            
        # Check consistency for each (i, k)
        W_prime = [[0] * rank for _ in range(pn)]
        ok = True
        for i in range(n):
            for k in range(p):
                # Build system for W'_{k,i,c}
                # sum_c U'_{i,j,c} V'_{l,k,c} W'_{k,i,c} = 8 * delta_{j,l}
                # M is m*m x rank
                M = np.zeros((m*m, rank), dtype=float)
                b = np.zeros(m*m, dtype=float)
                for j in range(m):
                    for l in range(m):
                        b[j*m+l] = 8.0 if j == l else 0.0
                        for c in range(rank):
                            M[j*m+l, c] = float(U_prime[i*m+j][c]) * float(V_prime[l*p+k][c])
                
                try:
                    sol, residuals, rank_np, sv = np.linalg.lstsq(M, b, rcond=None)
                    # Check if solution is close to integer
                    sol_int = np.round(sol)
                    if not np.allclose(M @ sol_int, b, atol=1e-6):
                        ok = False
                        break
                    # Store solution
                    W_prime[k*n+i] = [int(x) for x in sol_int]
                except:
                    ok = False
                    break
            if not ok:
                break
        
        if not ok:
            continue
        
        # Convert to half-integers
        U = [[x / 2 for x in row] for row in U_prime]
        V = [[x / 2 for x in row] for row in V_prime]
        W = [[x / 2 for x in row] for row in W_prime]
        
        # Final exact validation
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
    # Focus on ranks that are likely achievable: 35, 34, 33...
    ranks_to_try = [35, 34, 33, 32, 31, 30]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate more time to lower ranks
        if rank <= 33:
            max_iter = 50000
        else:
            max_iter = 100000
            
        rng = random.Random(rank * 7919 + 12345)
        result = _search_rank_integer(rank, n, m, p, deadline, rng, max_iter)
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
