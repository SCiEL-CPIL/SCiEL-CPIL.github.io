"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using the tensor equation."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_solve_for_W(U, V, r, n, m, p):
    """
    Given U (n*m x r) and V (m*p x r), try to find W (p*n x r)
    such that for all (i,j,k): sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] = 1.
    
    This is a system of n*m*p linear equations in p*n*r unknowns.
    We solve using Gaussian elimination over rationals.
    """
    num_eqs = n * m * p
    num_unknowns = p * n * r  # W flattened
    
    # Build the linear system: for each (i,j,k), equation is
    # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    # Unknown index for W[k*n+i][c] is (k*n+i)*r + c
    
    # Build augmented matrix as list of lists of Fractions
    # For efficiency, build sparse representation
    # Each equation has at most r non-zero entries
    
    # Map unknown index
    # W row index: k*n + i, column index: c -> flat index (k*n+i)*r + c
    
    # Build equations
    # eq_idx -> (list of (unknown_idx, coefficient), rhs)
    equations = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeffs = []
                for c in range(r):
                    val = Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
                    if val != 0:
                        unk_idx = (k * n + i) * r + c
                        coeffs.append((unk_idx, val))
                equations.append((coeffs, Fraction(1)))
    
    # Gaussian elimination
    # We'll work with a dense augmented matrix but only for the rows that have non-zero coeffs
    # Actually, let's use a more efficient approach:
    # Select r unknowns as "free" variables set to 0, and solve for the rest.
    # Better: use the full system with partial pivoting.
    
    # For small systems (n*m*p <= 30, p*n*r can be up to 30*2*5=300 for r=5),
    # let's build a dense matrix.
    
    # Actually, let's be smarter. The number of equations is n*m*p = 30.
    # The number of unknowns is p*n*r. For r=5, that's 50.
    # So it's underdetermined. We can set some variables to 0 and solve.
    
    # Strategy: pick a subset of unknowns to be "basic" (n*m*p of them if possible),
    # set the rest to 0, and solve.
    
    # Even simpler: since the system is underdetermined, let's try setting
    # W entries in a structured way.
    
    # Alternative: use numpy for speed, then verify with fractions.
    # Build the system matrix A (num_eqs x num_unknowns) and b (num_eqs)
    
    A = np.zeros((num_eqs, num_unknowns), dtype=float)
    b = np.ones(num_eqs, dtype=float)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    val = float(Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]))
                    if val != 0:
                        unk_idx = (k * n + i) * r + c
                        A[eq_idx, unk_idx] = val
                eq_idx += 1
    
    # Use least squares or lstsq to find a solution
    # Since underdetermined, we want a minimum-norm solution
    try:
        # Use lstsq which handles underdetermined systems
        sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        return None
    
    # Check if it's an exact solution
    A_sol = A @ sol
    if not np.allclose(A_sol, b, atol=1e-10):
        return None
    
    # Convert to fractions and check exactness
    W_flat = sol
    W = [[0] * r for _ in range(p * n)]
    for idx in range(p * n):
        for c in range(r):
            val = W_flat[idx * r + c]
            # Round to nearest half-integer
            rounded = round(val * 2) / 2
            if abs(val - rounded) > 1e-9:
                return None
            W[idx][c] = rounded
    
    # Verify with exact arithmetic
    if not _verify(U, V, W, n, m, p):
        return None
    
    return W

def _generate_structured_U_V(n, m, p, r, variant):
    """
    Generate structured U and V matrices for different patterns.
    """
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    
    if variant == 0:
        # Identity-like: U[i*m+j][c] = 1 if c == i*m+j (for c < n*m)
        for idx in range(n * m):
            if idx < r:
                U[idx][idx] = 1
        for idx in range(m * p):
            if idx < r:
                V[idx][idx] = 1
    elif variant == 1:
        # Cyclic: U[i*m+j][c] = 1 if (i*m+j) % r == c
        for idx in range(n * m):
            U[idx][idx % r] = 1
        for idx in range(m * p):
            V[idx][idx % r] = 1
    elif variant == 2:
        # Block: U[i*m+j][c] = 1 if j == c (for c < m)
        for i in range(n):
            for j in range(m):
                if j < r:
                    U[i * m + j][j] = 1
        for j in range(m):
            for k in range(p):
                if k < r:
                    V[j * p + k][k] = 1
    elif variant == 3:
        # Random with small integers, seeded
        import random
        rng = random.Random(variant * 1000 + r)
        for idx in range(n * m):
            for c in range(r):
                U[idx][c] = rng.choice([-1, 0, 1])
        for idx in range(m * p):
            for c in range(r):
                V[idx][c] = rng.choice([-1, 0, 1])
    elif variant == 4:
        # U uses i-index, V uses j-index
        for i in range(n):
            for j in range(m):
                if i < r:
                    U[i * m + j][i] = 1
        for j in range(m):
            for k in range(p):
                if j < r:
                    V[j * p + k][j] = 1
    elif variant == 5:
        # U uses i, V uses k
        for i in range(n):
            for j in range(m):
                if i < r:
                    U[i * m + j][i] = 1
        for j in range(m):
            for k in range(p):
                if k < r:
                    V[j * p + k][k] = 1
    elif variant == 6:
        # All ones in first column
        for idx in range(n * m):
            U[idx][0] = 1
        for idx in range(m * p):
            V[idx][0] = 1
        # Additional structure
        for i in range(n):
            for j in range(m):
                if j < r - 1:
                    U[i * m + j][j + 1] = 1
        for j in range(m):
            for k in range(p):
                if k < r - 1:
                    V[j * p + k][k + 1] = 1
    elif variant == 7:
        # Random with wider range
        import random
        rng = random.Random(variant * 1000 + r + 77)
        for idx in range(n * m):
            for c in range(r):
                U[idx][c] = rng.choice([-2, -1, 0, 1, 2])
        for idx in range(m * p):
            for c in range(r):
                V[idx][c] = rng.choice([-2, -1, 0, 1, 2])
    elif variant == 8:
        # U: i*m+j mapped to c via (i+c)%r, V: j*p+k mapped to c via (j+c)%r
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    if (i + c) % r == c and (j + c) % r == c:
                        U[i * m + j][c] = 1
        # Simplify: just use modular
        for i in range(n):
            for j in range(m):
                U[i * m + j][(i * m + j) % r] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][(j * p + k) % r] = 1
    elif variant == 9:
        # Strassen-like: use sums/differences
        # For n=2: U[0*m+j][0]=1, U[1*m+j][0]=1, U[0*m+j][1]=1, U[1*m+j][1]=-1
        # This gives (row0+row1) and (row0-row1) patterns
        for j in range(m):
            U[0 * m + j][0] = 1
            U[1 * m + j][0] = 1
            if r > 1:
                U[0 * m + j][1] = 1
                U[1 * m + j][1] = -1
            if r > 2:
                U[0 * m + j][2] = 1
                U[1 * m + j][2] = 1
        for k in range(p):
            V[0 * p + k][0] = 1
            V[1 * p + k][0] = 1
            V[2 * p + k][0] = 1
            if r > 1:
                V[0 * p + k][1] = 1
                V[1 * p + k][1] = 1
                V[2 * p + k][1] = 1
            if r > 2:
                V[0 * p + k][2] = 1
                V[1 * p + k][2] = 1
                V[2 * p + k][2] = 1
    elif variant == 10:
        # Kronecker product structure
        # U[i*m+j][c] where c = i*(m) + j mod r... just use direct
        for i in range(n):
            for j in range(m):
                U[i * m + j][i * m + j if i * m + j < r else 0] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][j * p + k if j * p + k < r else 0] = 1
    elif variant == 11:
        # Use Fraction-based random with half-integers
        import random
        rng = random.Random(variant * 1000 + r + 33)
        vals = [Fraction(x, 2) for x in range(-4, 5)]  # -2 to 2 in steps of 0.5
        for idx in range(n * m):
            for c in range(r):
                U[idx][c] = float(rng.choice(vals))
        for idx in range(m * p):
            for c in range(r):
                V[idx][c] = float(rng.choice(vals))
    elif variant == 12:
        # Sparse random: each row has exactly 1 non-zero
        import random
        rng = random.Random(variant * 1000 + r + 55)
        for idx in range(n * m):
            c = rng.randint(0, r - 1)
            U[idx][c] = rng.choice([-1, 1])
        for idx in range(m * p):
            c = rng.randint(0, r - 1)
            V[idx][c] = rng.choice([-1, 1])
    elif variant == 13:
        # U: each row has 2 non-zeros
        import random
        rng = random.Random(variant * 1000 + r + 66)
        for idx in range(n * m):
            cols = rng.sample(range(r), min(2, r))
            for c in cols:
                U[idx][c] = rng.choice([-1, 1])
        for idx in range(m * p):
            cols = rng.sample(range(r), min(2, r))
            for c in cols:
                V[idx][c] = rng.choice([-1, 1])
    elif variant == 14:
        # Toeplitz-like structure
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    if (i + j + c) % r == 0:
                        U[i * m + j][c] = 1
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    if (j + k + c) % r == 0:
                        V[j * p + k][c] = 1
    else:
        # Fallback: random
        import random
        rng = random.Random(variant * 1000 + r + variant)
        for idx in range(n * m):
            for c in range(r):
                U[idx][c] = rng.choice([-1, 0, 1])
        for idx in range(m * p):
            for c in range(r):
                V[idx][c] = rng.choice([-1, 0, 1])
    
    return U, V

def _solve_linear_system_rational(A_rows, b, num_unknowns):
    """
    Solve a system of linear equations over rationals using Gaussian elimination.
    A_rows: list of (list of (col, coeff), rhs)
    Returns solution as list of Fractions or None if no solution.
    """
    num_eqs = len(A_rows)
    
    # Convert to dense augmented matrix
    aug = [[Fraction(0)] * (num_unknowns + 1) for _ in range(num_eqs)]
    for eq_idx, (coeffs, rhs) in enumerate(A_rows):
        for col, coeff in coeffs:
            aug[eq_idx][col] = coeff
        aug[eq_idx][num_unknowns] = rhs
    
    # Gaussian elimination
    row = 0
    pivot_cols = []
    for col in range(num_unknowns):
        if row >= num_eqs:
            break
        # Find pivot
        pivot = -1
        for r in range(row, num_eqs):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        
        # Swap
        aug[row], aug[pivot] = aug[pivot], aug[row]
        pivot_cols.append(col)
        
        # Eliminate below
        pivot_val = aug[row][col]
        for r in range(row + 1, num_eqs):
            if aug[r][col] != 0:
                factor = aug[r][col] / pivot_val
                for c in range(col, num_unknowns + 1):
                    aug[r][c] -= factor * aug[row][c]
        
        row += 1
    
    # Check for inconsistency
    for r in range(row, num_eqs):
        if aug[r][num_unknowns] != 0:
            return None
    
    # Back substitution (for the pivoted variables)
    # Free variables set to 0
    sol = [Fraction(0)] * num_unknowns
    for i in range(row - 1, -1, -1):
        col = pivot_cols[i]
        total = aug[i][num_unknowns]
        for j in range(i + 1, row):
            total -= aug[i][pivot_cols[j]] * sol[pivot_cols[j]]
        sol[col] = total / aug[i][col]
    
    return sol

def _try_solve_for_W_rational(U, V, r, n, m, p):
    """
    Solve for W using exact rational arithmetic.
    """
    num_eqs = n * m * p
    num_unknowns = p * n * r
    
    # Build equations
    equations = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeffs = []
                for c in range(r):
                    val = Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c])
                    if val != 0:
                        unk_idx = (k * n + i) * r + c
                        coeffs.append((unk_idx, val))
                equations.append((coeffs, Fraction(1)))
    
    sol = _solve_linear_system_rational(equations, None, num_unknowns)
    if sol is None:
        return None
    
    # Check if solution is integer or half-integer
    W = [[0] * r for _ in range(p * n)]
    for idx in range(p * n):
        for c in range(r):
            val = sol[idx * r + c]
            # Check if it's integer or half-integer
            doubled = val * 2
            if doubled.denominator != 1:
                return None
            W[idx][c] = float(val)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        return None
    
    return W

def _random_search_with_solve(n, m, p, r, deadline, max_attempts=200):
    """
    Randomly generate U and V, then solve for W.
    """
    import random
    rng = random.Random(int(time.time() * 1000) % 100000)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Generate random U and V with small entries
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        
        W = _try_solve_for_W(U, V, r, n, m, p)
        if W is not None:
            return {"U": U, "V": V, "W": W}
    
    return None

def _structured_solve_search(n, m, p, r, deadline):
    """
    Try structured U and V patterns, then solve for W.
    """
    max_variants = 15
    for variant in range(max_variants):
        if time.time() > deadline:
            break
        
        U, V = _generate_structured_U_V(n, m, p, r, variant)
        W = _try_solve_for_W(U, V, r, n, m, p)
        if W is not None:
            return {"U": U, "V": V, "W": W}
    
    # Also try rational solve for a few variants
    for variant in range(min(5, max_variants)):
        if time.time() > deadline:
            break
        
        U, V = _generate_structured_U_V(n, m, p, r, variant)
        W = _try_solve_for_W_rational(U, V, r, n, m, p)
        if W is not None:
            return {"U": U, "V": V, "W": W}
    
    return None

def _direct_random_search(n, m, p, r, deadline, max_attempts=500):
    """
    Direct random search over all three matrices (fallback).
    """
    import random
    rng = random.Random(42)
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Deadline: leave 15 seconds margin before 150s limit
    deadline = time.time() + 130
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(10, 15, 6) = 15
    # Try ranks from 15 upward
    
    for r in range(15, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Strategy 1: Structured U,V + solve for W
        result = _structured_solve_search(n, m, p, r, deadline)
        if result is not None:
            best = result
            break
        
        # Strategy 2: Random U,V + solve for W
        result = _random_search_with_solve(n, m, p, r, deadline, max_attempts=100)
        if result is not None:
            best = result
            break
        
        # Strategy 3: Direct random search (less likely to succeed but fast)
        result = _direct_random_search(n, m, p, r, deadline, max_attempts=200)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END
