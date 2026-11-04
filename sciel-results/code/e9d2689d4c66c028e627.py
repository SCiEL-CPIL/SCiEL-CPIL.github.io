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

def _solve_w_system(U, V, i, k, n, m, p, rank):
    """Solve for W[k*n+i] given U, V. Returns list of Fractions or None if inconsistent."""
    # Build system: for each (j,l) in [0,m)x[0,m): sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] = delta_{j,l}
    # Note: l ranges over [0,m) not [0,p) because delta_{j,l} only makes sense for j,l in [0,m)
    # Wait, re-reading the constraint: sum_c U[i,j,c] V[l,k,c] W[k,i,c] = delta_{j,l}
    # Here j in [0,m), l in [0,m) (the inner dimension). But V is indexed by [j*p+k] where j in [0,m), k in [0,p)
    # So V[l*p+k] where l in [0,m), k in [0,p). The constraint is over j,l both in [0,m).
    
    rows = []
    rhs = []
    for j in range(m):
        for l in range(m):
            row = [Frac(U[i * m + j][c] * V[l * p + k][c]) for c in range(rank)]
            rows.append(row)
            rhs.append(Frac(1) if j == l else Frac(0))
    
    R = len(rows)  # m*m equations
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
    
    # Check consistency
    for rr in range(R):
        coeff_zero = all(aug[rr][c] == 0 for c in range(C))
        if coeff_zero and aug[rr][C] != 0:
            return None
    
    # Particular solution: free variables = 0
    Wvec = [Frac(0)] * C
    for (r, c) in pivots:
        Wvec[c] = aug[r][C]
    
    return Wvec

def _fast_consistency_check(U, V, i, k, n, m, p, rank):
    """Fast numpy-based consistency check for the W system."""
    # Build coefficient matrix as float
    M = np.zeros((m * m, rank), dtype=float)
    for j in range(m):
        for l in range(m):
            for c in range(rank):
                M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
    
    # RHS
    b = np.zeros(m * m, dtype=float)
    for j in range(m):
        b[j * m + j] = 1.0
    
    # Check consistency: rank([M|b]) == rank(M)
    M_aug = np.column_stack([M, b])
    rank_M = np.linalg.matrix_rank(M)
    rank_aug = np.linalg.matrix_rank(M_aug)
    if rank_M != rank_aug:
        return False
    return True

def _search_rank(rank, n, m, p, deadline, rng, max_iter):
    """Try to find an exact rank-`rank` decomposition. Returns (U,V,W) or None."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    # Use a wider range of integers to increase search space coverage
    # Optimal decompositions often use small integers like -2, -1, 0, 1, 2
    range_val = 2 
    if rank > 30:
        range_val = 1 # Try simpler first if rank is high? No, always try wider.
    range_val = 3 # Let's go with -3 to 3

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
                # Fast float check first
                if not _fast_consistency_check(U, V, i, k, n, m, p, rank):
                    ok = False
                    break
                
                # Pre-check with floats to see if W is likely half-integer
                M = np.zeros((m * m, rank), dtype=float)
                b = np.zeros(m * m, dtype=float)
                for j in range(m):
                    for l in range(m):
                        for c in range(rank):
                            M[j * m + l, c] = float(U[i * m + j][c]) * float(V[l * p + k][c])
                        b[j * m + j] = 1.0
                
                # Solve float system
                try:
                    W_float = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    # Singular, use exact solver to check consistency
                    Wvec = _solve_w_system(U, V, i, k, n, m, p, rank)
                    if Wvec is None:
                        ok = False
                        break
                    # Check if float solution matches exact? 
                    # If singular, there are multiple solutions. We pick one.
                    # The float solve might fail if singular. 
                    # In that case, we must use exact.
                    # But if it's singular, the solution is not unique.
                    # The exact solver picks a particular solution (free vars = 0).
                    # We should check if THAT solution is half-integer.
                    W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
                    continue

                # Check if W_float is close to half-integers
                W_half_int = np.round(W_float * 2) / 2
                if not np.allclose(W_float, W_half_int, atol=1e-9):
                    ok = False
                    break
                
                # If it looks like half-integers, verify with exact arithmetic
                Wvec = _solve_w_system(U, V, i, k, n, m, p, rank)
                if Wvec is None:
                    ok = False
                    break
                
                # Verify exact match with float prediction?
                # The exact solver might pick a different solution if the system is underdetermined.
                # But we checked consistency. 
                # If the system is full rank, the solution is unique.
                # If it's underdetermined, `np.linalg.solve` fails.
                # So if `np.linalg.solve` succeeded, the solution is unique.
                # So Wvec should match W_half_int.
                
                W[k * n + i] = [int(x) if x.denominator == 1 else x for x in Wvec]
            if not ok:
                break
        
        if not ok:
            continue
        
        # Check half-integer constraint
        if not _check_half_integer(W, rank):
            continue
        
        # Full exact verification
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
    
    # First try to find a better decomposition
    best_result = _schoolbook(n, m, p)
    
    # Try ranks in increasing order from the theoretical minimum up to schoolbook-1
    # Known bounds for 3x3x4 suggest rank 26 is optimal.
    # We will try 26, 27, 28... up to 35.
    # If we find one, we can stop or continue to see if we find lower (but we start from low).
    # Actually, if we find 26, that's likely the best.
    
    # Allocate time budget
    # Let's try 26 first with a generous budget.
    
    target_ranks = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    for rank in target_ranks:
        if time.time() > deadline - 10: # Stop if less than 10s left
            break
            
        rng = random.Random(rank * 12345 + 678)
        # Increase iterations for lower ranks as they are harder? Or same?
        # Lower ranks are harder, so maybe give them more time.
        # But total time is fixed.
        # Let's just run until deadline for each rank? No, that would starve higher ranks.
        # Let's give each rank a slice of the remaining time.
        
        start_rank_time = time.time()
        # Give 15 seconds per rank attempt? 
        # 10 ranks * 15s = 150s. Too much.
        # Let's give 10 seconds per rank.
        rank_deadline = min(deadline - 5, start_rank_time + 10)
        
        try:
            result = _search_rank(rank, n, m, p, rank_deadline, rng, max_iter=100000)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                # If we found a rank significantly lower than schoolbook, we can stop.
                # 26 is optimal. 27 is very good.
                if rank < 30:
                    break
        except Exception:
            pass

    return best_result
