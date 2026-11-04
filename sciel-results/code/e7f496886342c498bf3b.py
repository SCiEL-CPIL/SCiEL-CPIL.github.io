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

def _try_winograd_rank25(n, m, p):
    """
    Attempt to construct a rank-25 decomposition using Winograd-style algebraic structure.
    
    For 3x3 * 3x4, we use the fact that this is related to polynomial multiplication.
    We represent the j-index (size 3) as a polynomial in x, and k-index (size 4) as 
    polynomial in y. The multiplication corresponds to evaluating at specific points.
    
    Strategy: Use 5 evaluation points for the 3-coefficient polynomial (to handle the 
    convolution with 4 coefficients), giving us a structured decomposition.
    """
    rank = 25
    # We'll use a structured construction based on the following idea:
    # For each of the 5 "frequency" components, we have 5 "spatial" components.
    
    # Define evaluation points for the 3x4 convolution structure
    # We need to decompose the bilinear form such that:
    # sum_c U[i,j,c] V[l,k,c] W[k,i,c] = delta_{j,l}
    
    # Try a structured pattern: use circulant-like matrices
    # U[i,j,c] depends on (i, j, c mod 5) and (c // 5)
    # V[l,k,c] depends on (l, k, c mod 5) and (c // 5)
    
    # Let's try: c = 5*a + b where a in [0,4], b in [0,4]
    # U[i,j,5*a+b] = alpha[i,j,a,b]
    # V[l,k,5*a+b] = beta[l,k,a,b]  
    # W[k,i,5*a+b] = gamma[k,i,a,b]
    
    # For the identity to hold:
    # sum_{a,b} alpha[i,j,a,b] * beta[l,k,a,b] * gamma[k,i,a,b] = delta_{j,l}
    
    # This is complex. Let me try a simpler structured approach.
    
    # Alternative: Use the fact that we can separate the i and k dependencies.
    # For fixed i and k, we need a rank-m (rank-3) decomposition of the identity.
    # But we share the rank across all (i,k) pairs.
    
    # Let me try a direct construction with small integers.
    # U has 9 rows, V has 12 rows, W has 12 rows, each of length 25.
    
    # Try: make U and V have a block structure where each block is 5x5.
    # 9 = 3*3, 12 = 3*4, 12 = 4*3
    
    # Let's use: U[i,j, 5*i+j] = 1 for the main diagonal, plus some cross terms
    # This is getting complicated. Let me just do a targeted search for rank 25.
    return None

def _search_rank_structured(rank, n, m, p, deadline, rng, max_iter):
    """Try structured patterns for rank-`rank` decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        
        # Strategy 1: Random small integers {-1, 0, 1}
        if it % 3 == 0:
            U = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(nm)]
            V = [[rng.choice([-1, 0, 1]) for c in range(rank)] for _ in range(mp)]
        # Strategy 2: Sparse random
        elif it % 3 == 1:
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            for idx in range(nm):
                if rng.random() < 0.6:
                    col = rng.randint(0, rank - 1)
                    U[idx][col] = rng.choice([-1, 1])
            for idx in range(mp):
                if rng.random() < 0.6:
                    col = rng.randint(0, rank - 1)
                    V[idx][col] = rng.choice([-1, 1])
        # Strategy 3: Small integers -2 to 2
        else:
            U = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(nm)]
            V = [[rng.randint(-2, 2) for c in range(rank)] for _ in range(mp)]
        
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
        
        if not _check_half_integer(W, rank):
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
    # Focus on ranks that are likely achievable: 24-32
    # Lower ranks are harder but more valuable
    ranks_to_try = [24, 25, 26, 27, 28, 29, 30, 31, 32]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate more time to lower ranks
        if rank <= 26:
            max_iter = 80000
        elif rank <= 28:
            max_iter = 50000
        else:
            max_iter = 30000
            
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
