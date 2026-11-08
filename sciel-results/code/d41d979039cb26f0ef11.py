"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction
from collections import defaultdict

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if s != 1:
                    return False
    return True


def _solve_linear_exact(A, b):
    """Solve A x = b exactly using fraction arithmetic via Gaussian elimination.
    A is list of lists (m_eq x n_var), b is list of numbers (m_eq).
    Returns x (list of Fraction, length n_var) or None if inconsistent.
    If underdetermined, sets free variables to 0."""
    m_eq = len(A)
    if m_eq == 0:
        n_var = len(A[0]) if A else 0
        return [Fraction(0)] * n_var
    n_var = len(A[0])
    
    # Build augmented matrix with Fractions
    aug = []
    for i in range(m_eq):
        row = [Fraction(A[i][j]) for j in range(n_var)] + [Fraction(b[i])]
        aug.append(row)
    
    rank = 0
    pivot_cols = []
    for col in range(n_var):
        # Find pivot
        pivot_row = None
        for row in range(rank, m_eq):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        aug[rank], aug[pivot_row] = aug[pivot_row], aug[rank]
        pivot_cols.append(col)
        # Eliminate all other rows
        pivot_val = aug[rank][col]
        for row in range(m_eq):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col] / pivot_val
                for j in range(col, n_var + 1):
                    aug[row][j] -= factor * aug[rank][j]
        rank += 1
        if rank == n_var:
            break
    
    # Check consistency
    for row in range(rank, m_eq):
        if aug[row][n_var] != 0:
            return None
    
    # Extract solution: free variables = 0
    x = [Fraction(0)] * n_var
    for i, pc in enumerate(pivot_cols):
        x[pc] = aug[i][n_var] / aug[i][pc]
    
    # Verify
    for i in range(m_eq):
        s = sum(A[i][j] * x[j] for j in range(n_var))
        if s != b[i]:
            return None
    
    return x


def _try_slice_method(r, U, V, n, m, p, deadline):
    """Given U (n*m x r) and V (m*p x r), solve for W (p*n x r).
    For each (k, i), solve: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j.
    Returns W or None."""
    W = [[Fraction(0)] * r for _ in range(p * n)]
    
    for k in range(p):
        for i in range(n):
            # Build system: for each j, sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            A = []
            for j in range(m):
                row = []
                for c in range(r):
                    row.append(U[i * m + j][c] * V[j * p + k][c])
                A.append(row)
            b = [1] * m
            
            sol = _solve_linear_exact(A, b)
            if sol is None:
                return None
            W[k * n + i] = sol
        
        if time.time() > deadline:
            return None
    
    return W


def _is_half_integer(x):
    """Check if x is an integer or half-integer (i.e., 2*x is integer)."""
    x2 = x * 2
    return x2.denominator == 1


def _try_rank(r, deadline, n, m, p):
    """Try to find a rank-r decomposition. Returns dict or None."""
    # Strategy: try structured U and V patterns
    # U is (n*m) x r = 9 x r
    # V is (m*p) x r = 15 x r
    
    # Pattern: each row of U has a few 1s, each row of V has a few 1s
    # We'll try: U rows are unit vectors or sums of 2 unit vectors
    # V rows similarly
    
    # For efficiency, try U where each row has exactly 1 or 2 non-zero entries (value 1)
    # and V similarly. Then solve for W.
    
    # Number of possible rows for U: C(r,1) + C(r,2) = r + r*(r-1)/2
    # For r=15: 15 + 105 = 120 possible rows per U row
    # 9 rows of U: 120^9 is too large
    
    # Better: try specific structured patterns
    # Pattern A: U[i*m+j, c] = 1 if c == i*m+j (identity-like for first n*m columns)
    #            and 0 otherwise, if r >= n*m
    # Then V[j*p+k, c] needs to be set so that for each (i,j,k):
    #   U[i*m+j, c] * V[j*p+k, c] is nonzero only when c = i*m+j
    #   So V[j*p+k, i*m+j] must be 1 for the product to contribute
    #   But V[j*p+k, c] is the same for all i that share j... 
    #   V[j*p+k, i*m+j] depends on i, but V row is indexed by (j,k) only.
    #   So V[j*p+k, i*m+j] can only be one value for fixed (j,k).
    #   This means we can't have V[j*p+k, i*m+j] = 1 for all i simultaneously
    #   unless all i*m+j are the same column, which they're not.
    
    # Let me try a different approach:
    # Use the fact that we can have U and V with overlapping support patterns.
    
    # Try: U[i*m+j, c] = 1 if c == j*p+k for some k (linking U to V via j)
    # Actually, let's try a more systematic search for small r.
    
    # For r <= 20, try all U where each row has exactly 1 non-zero entry (value 1)
    # That gives r choices per row, r^9 total for U. For r=15, 15^9 ~ 3.8e10, too large.
    
    # Try r=15 with U rows being unit vectors, but use symmetry breaking:
    # Fix U[0] = e_0, U[1] = e_1, ... to break symmetry? No, that's too restrictive.
    
    # Alternative: use the known lower bound and try to construct.
    # Lower bound for matmul rank: max(n*p, m*p, n*m) = max(15, 15, 9) = 15
    # But actual rank might be higher.
    
    # Let me try a randomized search with exact verification for r = 15, 16, 17, 18
    # For each trial, generate random sparse U, V with entries in {0, 1}
    # and check if W can be solved with half-integer entries.
    
    import random
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        # Generate random U: 9 x r, each row has 1-3 ones
        U = []
        for row_idx in range(n * m):
            row = [0] * r
            num_ones = random.choice([1, 1, 2, 2, 3])
            cols = random.sample(range(r), min(num_ones, r))
            for c in cols:
                row[c] = 1
            U.append(row)
        
        # Generate random V: 15 x r, each row has 1-3 ones
        V = []
        for row_idx in range(m * p):
            row = [0] * r
            num_ones = random.choice([1, 1, 2, 2, 3])
            cols = random.sample(range(r), min(num_ones, r))
            for c in cols:
                row[c] = 1
            V.append(row)
        
        # Try to solve for W
        W = _try_slice_method(r, U, V, n, m, p, deadline)
        if W is None:
            continue
        
        # Check if W has half-integer entries
        valid = True
        for row in W:
            for val in row:
                if not _is_half_integer(val):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Convert to int or half-int representation
            U_out = [[1 if x else 0 for x in row] for row in U]
            V_out = [[1 if x else 0 for x in row] for row in V]
            W_out = []
            for row in W:
                wrow = []
                for val in row:
                    if val.denominator == 1:
                        wrow.append(int(val))
                    else:
                        wrow.append(float(val))
                W_out.append(wrow)
            
            # Verify
            if _verify(U_out, V_out, W_out, n, m, p):
                return {"U": U_out, "V": V_out, "W": W_out}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*p, m*p, n*m) = 15
    # Schoolbook: 45
    
    for r in range(15, 45):
        if time.time() > deadline - 5:
            break
        result = _try_rank(r, deadline, n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
