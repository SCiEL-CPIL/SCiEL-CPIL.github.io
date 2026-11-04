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

def _try_structured_search(rank, n, m, p, deadline, rng, max_iter=10000):
    """Try structured U,V and check if W can be solved."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Generate random U and V with small integers
        U = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(nm)]
        V = [[rng.randint(-1, 1) for c in range(rank)] for _ in range(mp)]
        
        # Quick check: no all-zero rows
        if any(all(x == 0 for x in row) for row in U):
            continue
        if any(all(x == 0 for x in row) for row in V):
            continue
        
        # Try to solve for all W[k*n+i]
        W = [[0] * rank for _ in range(pn)]
        ok = True
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
        
        if not _check_half_integer(W, rank):
            continue
        
        if _validate(U, V, W, n, m, p, rank):
            return U, V, W
    
    return None

def _try_systematic_ranks(n, m, p, deadline):
    """Try ranks from low to high with increasing time allocation."""
    ranks_to_try = list(range(20, 37))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate more time to lower ranks
        if rank <= 25:
            max_iter = 5000
        else:
            max_iter = 2000
        
        # Strategy: Structured search
        rng = random.Random(rank * 1000 + 42)
        result = _try_structured_search(rank, n, m, p, deadline, rng, max_iter=max_iter)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_numpy_search(rank, n, m, p, deadline, rng, max_iter=20000):
    """Use numpy for faster linear algebra solving."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the constraint structure
    # For each (i, k), we have m*m equations
    # Equation (j, l): sum_c U[i*m+j][c] * V[l*p+k][c] * W[k*n+i][c] = delta(j,l)
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Generate random U and V
        U = np.array([[rng.randint(-2, 2) for c in range(rank)] for _ in range(nm)], dtype=float)
        V = np.array([[rng.randint(-2, 2) for c in range(rank)] for _ in range(mp)], dtype=float)
        
        # Check for all-zero rows
        if np.any(np.all(U == 0, axis=1)):
            continue
        if np.any(np.all(V == 0, axis=1)):
            continue
        
        # Solve for W
        W = np.zeros((pn, rank), dtype=float)
        ok = True
        
        for i in range(n):
            for k in range(p):
                # Build the system: m*m equations, rank unknowns
                # A[row_idx, c] = U[i*m+j][c] * V[l*p+k][c]
                # b[row_idx] = delta(j,l)
                A = np.zeros((m*m, rank))
                b = np.zeros(m*m)
                
                idx = 0
                for j in range(m):
                    for l in range(m):
                        A[idx] = U[i*m+j] * V[l*p+k]
                        b[idx] = 1.0 if j == l else 0.0
                        idx += 1
                
                # Solve using least squares, then check if it's exact
                try:
                    Wvec, residuals, rank_svd, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                
                # Check if the solution is exact
                if np.linalg.norm(A @ Wvec - b) > 1e-8:
                    ok = False
                    break
                
                # Check if entries are close to half-integers
                Wvec_rounded = np.round(Wvec * 2) / 2
                if np.linalg.norm(Wvec - Wvec_rounded) > 1e-8:
                    ok = False
                    break
                
                W[k*n+i] = Wvec_rounded
            
            if not ok:
                break
        
        if not ok:
            continue
        
        # Convert to exact fractions for validation
        U_frac = [[Frac(int(U[i][c])) for c in range(rank)] for i in range(nm)]
        V_frac = [[Frac(int(V[i][c])) for c in range(rank)] for i in range(mp)]
        W_frac = [[Frac(int(W[i][c] * 2)) / 2 for c in range(rank)] for i in range(pn)]
        
        if _validate(U_frac, V_frac, W_frac, n, m, p, rank):
            return U_frac, V_frac, W_frac
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    
    # Try systematic search for lower rank
    try:
        result = _try_systematic_ranks(n, m, p, deadline)
        if result is not None:
            best_result = result
    except Exception:
        pass
    
    # If still schoolbook, try numpy-based search
    if best_result['U'][0].count(0) == 35:  # Still rank 36
        try:
            for rank in range(20, 30):
                if time.time() > deadline:
                    break
                rng = random.Random(rank * 777 + 123)
                result = _try_numpy_search(rank, n, m, p, deadline, rng, max_iter=5000)
                if result is not None:
                    U, V, W = result
                    best_result = {"U": U, "V": V, "W": W}
                    break
        except Exception:
            pass
    
    return best_result
