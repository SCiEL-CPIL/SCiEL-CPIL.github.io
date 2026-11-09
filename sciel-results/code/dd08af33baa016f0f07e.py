"""Program that searches for a rank-minimal exact tensor decomposition
of the matrix multiplication tensor for C(3,5) * A(5,7) = B(3,7)."""

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
from fractions import Fraction
import sys

def _gaussian_eliminate(rows):
    """
    Perform Gaussian elimination over the rationals.
    rows: list of lists of Fraction
    Returns (rank, pivots) where pivots is list of (row_idx, col_idx) for each pivot.
    Also modifies rows in-place to RREF.
    """
    if not rows:
        return 0, []
    nrows = len(rows)
    ncols = len(rows[0]) if nrows > 0 else 0
    rank = 0
    pivots = []
    
    for col in range(ncols):
        # Find pivot row
        pivot_row = -1
        for r in range(rank, nrows):
            if rows[r][col] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        
        # Swap
        rows[rank], rows[pivot_row] = rows[pivot_row], rows[rank]
        
        # Normalize
        pv = rows[rank][col]
        for c in range(col, ncols):
            rows[rank][c] = rows[rank][c] / pv
        
        # Eliminate
        for r in range(nrows):
            if r != rank and rows[r][col] != 0:
                factor = rows[r][col]
                for c in range(col, ncols):
                    rows[r][c] = rows[r][c] - factor * rows[rank][c]
        
        pivots.append((rank, col))
        rank += 1
        if rank == nrows:
            break
    
    return rank, pivots


def _solve_for_W(U, V, n, m, p, r):
    """
    Given U (n*m x r) and V (m*p x r) as lists of lists of Fraction,
    try to find W (p*n x r) such that for all (i,j,k):
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    
    This is linear in W. Build the system and solve.
    Returns W as list of lists of Fraction, or None if no solution.
    """
    # Build system: for each (i,j,k), we have one equation in variables W[0..p*n-1, 0..r-1]
    # Variable index: var = (k*n+i)*r + c  for W[k*n+i][c]
    # Equation for (i,j,k): sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
    
    # Total variables: p*n*r
    # Total equations: n*m*p
    num_vars = p * n * r
    num_eqs = n * m * p
    
    # Build augmented matrix
    # We'll store as list of lists, each row has num_vars+1 entries
    # But this could be large. Let's be smart.
    
    # For each equation (i,j,k), the coefficients are:
    # For variable (k'n + i')*r + c, coefficient is 0 unless (k',i') == (k,i)
    # So for equation (i,j,k), only variables (k*n+i)*r + c for c=0..r-1 appear,
    # with coefficient U[i*m+j][c] * V[j*p+k][c]
    
    # We can build a sparse-like structure but let's just build dense for small sizes.
    # n=3, m=5, p=7: num_eqs = 105, num_vars = 7*3*r = 21r
    # For r=15: num_vars=315, total entries ~ 105*316 = 33180, manageable.
    
    A = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = [Fraction(0)] * (num_vars + 1)
                row[-1] = Fraction(1)
                base = (k * n + i) * r
                for c in range(r):
                    coeff = U[i * m + j][c] * V[j * p + k][c]
                    if coeff != 0:
                        row[base + c] = coeff
                A.append(row)
    
    rank, pivots = _gaussian_eliminate(A)
    
    # Check consistency: if any row has all zeros in coefficients but nonzero in RHS
    for row in A:
        all_zero = all(row[c] == 0 for c in range(num_vars))
        if all_zero and row[-1] != 0:
            return None
    
    # Extract solution for W
    W = [[Fraction(0)] * r for _ in range(p * n)]
    pivot_cols = {pivots[i][1] for i in range(rank)}
    
    for row_idx in range(rank):
        col = pivots[row_idx][1]
        val = A[row_idx][-1]
        # col = (k*n+i)*r + c
        kni = col // r
        c = col % r
        W[kni][c] = val
    
    return W


def construct():
    n, m, p = DIMENSIONS
    
    # Lower bound: rank >= max(n*m, m*p, p*n) is not true generally.
    # But rank >= max(n, m, p) * min(n, p) / something...
    # Actually, a simple lower bound is rank >= n*m + m*p + p*n - 2*(n*m*p - something)
    # Let's just start from a reasonable lower bound.
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is NOT correct.
    # Strassen shows rank can be less than n*m*p.
    # A known lower bound is rank >= 2*max(n,p) + m - 1 or similar.
    # Let's try starting from r = 15 and going up.
    
    import random
    
    # We'll try random U, V and see if W exists
    # For efficiency, let's try a structured approach first.
    
    # Actually, let's try a systematic search with small r.
    # For each r, try random initializations of U and V, then solve for W.
    
    best_r = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # First, try the schoolbook decomposition to have a fallback
    schoolbook_r = n * m * p
    U_sb = [[Fraction(0)] * schoolbook_r for _ in range(n * m)]
    V_sb = [[Fraction(0)] * schoolbook_r for _ in range(m * p)]
    W_sb = [[Fraction(0)] * schoolbook_r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = Fraction(1)
                V_sb[j * p + k][col] = Fraction(1)
                W_sb[k * n + i][col] = Fraction(1)
                col += 1
    
    best_r = schoolbook_r
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Now try to find lower rank decompositions
    # We'll try random U, V for various r
    # Use a deterministic seed for reproducibility
    random.seed(42)
    
    max_r_to_try = min(schoolbook_r, 60)  # Don't try too many
    start_r = 10
    
    for r in range(start_r, max_r_to_try + 1):
        found = False
        # Try several random initializations
        for trial in range(20):
            # Generate random U and V with small integer entries
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            
            # Convert to Fraction
            U = [[Fraction(x) for x in row] for row in U]
            V = [[Fraction(x) for x in row] for row in V]
            
            W = _solve_for_W(U, V, n, m, p, r)
            if W is not None:
                # Verify the decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = Fraction(0)
                            for c in range(r):
                                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            if s != Fraction(1):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_r = r
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            # We found a decomposition with rank r, try lower? No, we're going up.
            # Actually we should try all r and keep the minimum.
            # But if we found one at r, we can stop trying higher r.
            break
    
    # Convert to plain lists of int/Fraction for output
    def convert(mat):
        out = []
        for row in mat:
            out_row = []
            for val in row:
                if val.denominator == 1:
                    out_row.append(val.numerator)
                else:
                    out_row.append(val)
            out.append(out_row)
        return out
    
    return {
        "U": convert(best_U),
        "V": convert(best_V),
        "W": convert(best_W)
    }
# EVOLVE-BLOCK-END
