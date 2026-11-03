"""Matrix multiplication 4,4,6 — search for a rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 6]
N, M, P = DIMENSIONS

# ---------------------------------------------------------------------------
# Tensor layout:  T[i, j, k] = 1  iff  i < N, j < M, k < P
# Index the entries by (i, j, k).  We will build a linear system
#     sum_c  U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  ==  T[i,j,k]
# which is linear in the rank-1 factors once we fix the support pattern.
# ---------------------------------------------------------------------------

TENSOR = {}
for i in range(N):
    for j in range(M):
        for k in range(P):
            TENSOR[(i, j, k)] = 1

TENSOR_LIST = list(TENSOR.keys())          # 96 entries
TENSOR_VALUES = [1] * len(TENSOR_LIST)

# ---------------------------------------------------------------------------
# A rank-1 "term" is specified by the sets of (i,j), (j,k), (k,i) it touches.
# The product term's contribution to entry (i,j,k) is
#     a[i,j] * b[j,k] * c[k,i]
# We fix the *support* (which a,b,c are nonzero) and solve for the values.
# ---------------------------------------------------------------------------

def _solve_term(support, rhs):
    """
    Given a support pattern (a dict mapping (i,j)->1, (j,k)->1, (k,i)->1
    encoded as three sets), solve for the unknowns a[i,j], b[j,k], c[k,i]
    such that for every entry in rhs,  a*b*c == rhs value.

    We linearize by fixing one variable per term to 1 (gauge) and solving
    the remaining bilinear system.  Simpler approach: treat all a,b,c as
    unknowns and solve the *linear* system obtained by fixing the product
    structure.  Actually the cleanest exact approach:

    For a fixed support, the term contributes to entries (i,j,k) where
    (i,j) in A, (j,k) in B, (k,i) in C.  The value at (i,j,k) is
    a[i,j]*b[j,k]*c[k,i].  This is still nonlinear in the unknowns.

    Instead we use the standard trick: the multiplication tensor
    decomposition is linear in the *W* matrix when U and V are fixed,
    and linear in U when V and W are fixed, etc.  We'll do alternating
    linear solves.
    """
    pass

# ---------------------------------------------------------------------------
# Cleaner formulation:
# A rank-r decomposition is three matrices U (nm x r), V (mp x r), W (pn x r)
# such that for all (i,j,k):
#     sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  ==  T[i,j,k]
#
# This is a system of 96 equations in 3*r variables, but it's *cubic*
# (product of three variables).  However, if we fix U and V, it's linear
# in W.  So we can do:
#   1. Randomly initialize U, V (sparse, integer/half-integer).
#   2. Solve for W linearly.
#   3. Verify.  If not exact, do local search on U/V entries.
#
# Even simpler and fully exact: use the fact that the tensor is 0-1.
# We can use a *randomized* construction: pick r random rank-1 terms
# (each term picks a random (i,j), (j,k), (k,i) and sets the three
# entries to 1), then solve a linear system for scaling factors.
#
# The most reliable exact method: build the full linear system where the
# unknowns are the r columns, but the equations are bilinear.  We handle
# this by *fixing one factor's entries* and solving the rest linearly.
# ---------------------------------------------------------------------------

# Let's use a different, well-known exact approach:
# Represent each rank-1 term by its three "index triples."  The product
# tensor of term c is the outer product of three vectors.  The sum of r
# such outer products must equal the target tensor.
#
# We'll use a *Gaussian elimination on the linearized system*:
# Fix U and V to specific sparse patterns.  Then the equations become
# linear in W.  Solve for W.  Check if W has the right structure
# (i.e., the decomposition is valid).  If not, perturb U/V and retry.

def _build_linear_system_for_W(U, V):
    """
    Given U (nm x r) and V (mp x r), build the linear system A_w * W_flat = b
    where W is (pn x r) flattened, and the constraint is
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == T[i,j,k].
    For fixed U, V, this is linear in W.
    """
    nm, mp, pn = N * M, M * P, P * N
    r = U[0].__len__() if U else 0
    # Unknowns: W[k*n+i, c] for k in range(P), i in range(N), c in range(r)
    # Total unknowns: pn * r
    # Equations: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # So each equation involves only the W entries for that (k,i) and all c.
    # Equation for (i,j,k): sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # This means for each (i,k), the equation is:
    #   sum_c coeff[c] * W[k*n+i, c] = 1
    # where coeff[c] = U[i*m+j,c] * V[j*p+k,c].
    # But wait — different j give different coeff vectors for the same (i,k)!
    # So the system is: for each (i,j,k), we have an equation.  Multiple (i,j,k)
    # with the same (i,k) but different j give *different* equations for the
    # same set of W unknowns.  This is a standard linear system.
    
    # Let's build it properly.
    # Unknown index: w_idx(k, i, c) = (k * N + i) * r + c
    # Equation for (i, j, k): sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] = 1
    
    # We'll collect equations as (list of (unknown_idx, coeff), rhs)
    equations = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                eq = []
                for c in range(r):
                    u_val = U[i * M + j][c]
                    v_val = V[j * P + k][c]
                    if u_val == 0 or v_val == 0:
                        continue
                    coeff = u_val * v_val
                    if coeff != 0:
                        widx = (k * N + i) * r + c
                        eq.append((widx, coeff))
                equations.append((eq, 1))
    
    # Now solve this linear system using Gaussian elimination over Fractions.
    num_unknowns = pn * r
    # Build augmented matrix
    # We'll do sparse-ish Gaussian elimination.
    
    # For efficiency, use a dict-based approach for the matrix rows.
    # Each row: dict mapping unknown_idx -> coeff, plus rhs.
    rows = []
    for eq, rhs in equations:
        row = {}
        for idx, coeff in eq:
            row[idx] = row.get(idx, 0) + coeff
        rows.append((row, rhs))
    
    # Gaussian elimination
    pivot_cols = []
    current_row = 0
    for col in range(num_unknowns):
        # Find pivot
        pivot = None
        for r_idx in range(current_row, len(rows)):
            row = rows[r_idx][0]
            if col in row and row[col] != 0:
                pivot = r_idx
                break
        if pivot is None:
            continue
        # Swap
        rows[current_row], rows[pivot] = rows[pivot], rows[current_row]
        pivot_cols.append(col)
        # Normalize
        pval = rows[current_row][0][col]
        if pval != 1:
            rows[current_row][0] = {k: v / pval for k, v in rows[current_row][0].items()}
            rows[current_row][1] = rows[current_row][1] / pval
        # Eliminate
        for r_idx in range(len(rows)):
            if r_idx == current_row:
                continue
            row = rows[r_idx][0]
            if col in row and row[col] != 0:
                factor = row[col]
                new_row = dict(row)
                for k, v in rows[current_row][0].items():
                    new_row[k] = new_row.get(k, 0) - factor * v
                if new_row.get(col, 0) == 0:
                    del new_row[col]
                rows[r_idx] = (new_row, rows[r_idx][1] - factor * rows[current_row][1])
        current_row += 1
        if current_row >= len(rows):
            break
    
    # Extract solution
    solution = [0] * num_unknowns
    # For each pivot column, the solution is in the corresponding row
    for idx, col in enumerate(pivot_cols):
        row, rhs = rows[idx]
        solution[col] = rhs
    
    # Check consistency: every non-pivot row must have rhs == 0
    for idx in range(len(pivot_cols), len(rows)):
        row, rhs = rows[idx]
        if rhs != 0:
            return None  # Inconsistent
        # Also check that the row is all zeros
        for k, v in row.items():
            if v != 0:
                return None
    
    return solution


def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the tensor."""
    nm, mp, pn = N * M, M * P, P * N
    r = len(U[0]) if U else 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0
                for c in range(r):
                    u_val = U[i * M + j][c]
                    v_val = V[j * P + k][c]
                    w_val = W[k * N + i][c]
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def _is_valid_entries(mat):
    """Check that all entries are integers or half-integers (denominator 1 or 2)."""
    for row in mat:
        for val in row:
            if isinstance(val, Fraction):
                if val.denominator not in (1, 2):
                    return False
            elif isinstance(val, (int, float)):
                if isinstance(val, float):
                    if val != int(val) and val * 2 != int(val * 2):
                        return False
            else:
                return False
    return True


def _random_sparse_matrix(rows, cols, density, rng):
    """Generate a random sparse matrix with entries in {0, 1/2, 1, 3/2, 2}."""
    mat = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if rng.random() < density:
                mat[i][j] = rng.choice([Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2)])
    return mat


def _local_search(U, V, W, rank, rng, time_limit):
    """
    Try to fix a near-miss decomposition by swapping individual entries.
    Returns (U, V, W, success) or (None, None, None, False).
    """
    start = time.time()
    best_residual = None
    best_U, best_V, best_W = U, V, W
    
    for _ in range(200):
        if time.time() - start > time_limit:
            break
        
        # Compute residual
        residual = {}
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    total = 0
                    for c in range(rank):
                        total += U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c]
                    residual[(i, j, k)] = total - 1
        
        if all(v == 0 for v in residual.values()):
            return U, V, W, True
        
        # Pick a random nonzero residual entry
        nonzero = [(key, val) for key, val in residual.items() if val != 0]
        if not nonzero:
            break
        (target_i, target_j, target_k), target_val = rng.choice(nonzero)
        
        # Try changing a random entry in U, V, or W
        factor = rng.choice(['U', 'V', 'W'])
        if factor == 'U':
            row = target_i * M + target_j
            col = rng.randrange(rank)
            old = U[row][col]
            new = old + rng.choice([Fraction(-1, 2), Fraction(1, 2), -1, 1])
            U[row][col] = new
        elif factor == 'V':
            row = target_j * P + target_k
            col = rng.randrange(rank)
            old = V[row][col]
            new = old + rng.choice([Fraction(-1, 2), Fraction(1, 2), -1, 1])
            V[row][col] = new
        else:
            row = target_k * N + target_i
            col = rng.randrange(rank)
            old = W[row][col]
            new = old + rng.choice([Fraction(-1, 2), Fraction(1, 2), -1, 1])
            W[row][col] = new
        
        # Check if improvement
        new_residual_sum = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    total = 0
                    for c in range(rank):
                        total += U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c]
                    new_residual_sum += abs(total - 1)
        
        if best_residual is None or new_residual_sum < best_residual:
            best_residual = new_residual_sum
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        else:
            # Revert
            if factor == 'U':
                U[row][col] = old
            elif factor == 'V':
                V[row][col] = old
            else:
                W[row][col] = old
    
    return best_U, best_V, best_W, False


def construct():
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    nm, mp, pn = N * M, M * P, P * N
    schoolbook_rank = N * M * P  # 96
    
    # Start with the schoolbook decomposition as fallback
    best_rank = schoolbook_rank
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                best_U[i * M + j][col] = 1
                best_V[j * P + k][col] = 1
                best_W[k * N + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # Strategy: for decreasing rank r, try random U, V patterns and solve for W.
    # If a valid W is found, verify and return.
    
    for r in range(80, schoolbook_rank):  # Try ranks from 80 to 95
        if time.time() > deadline:
            break
        
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random sparse U and V
            density = 0.3
            U = _random_sparse_matrix(nm, r, density, rng)
            V = _random_sparse_matrix(mp, r, density, rng)
            
            # Solve for W
            W_flat = _build_linear_system_for_W(U, V)
            if W_flat is None:
                continue
            
            # Reshape W_flat into W
            W = [[0] * r for _ in range(pn)]
            for k in range(P):
                for i in range(N):
                    for c in range(r):
                        W[k * N + i][c] = W_flat[(k * N + i) * r + c]
            
            # Verify
            if _verify(U, V, W):
                if _is_valid_entries(U) and _is_valid_entries(V) and _is_valid_entries(W):
                    return {"U": U, "V": V, "W": W}
            
            # Try local search to fix near-misses
            U2, V2, W2, success = _local_search(U, V, W, r, rng, 0.5)
            if success:
                if _is_valid_entries(U2) and _is_valid_entries(V2) and _is_valid_entries(W2):
                    return {"U": U2, "V": V2, "W": W2}
    
    # If we ran out of time, return the best we have (schoolbook)
    return {"U": best_U, "V": best_V, "W": best_W}
