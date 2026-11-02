import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]
DEADLINE = 138.0  # seconds; leave 12s margin before the 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
    T = np.einsum('at,bt,ct->abc', U, V, W)
    # Expected: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                    return False
    # All other entries should be 0
    expected_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected_mask[i*m+j, j*p+k, k*n+i] = True
    if np.max(np.abs(T[~expected_mask])) > tol:
        return False
    return True

def _build_rank_55():
    """Build the rank-55 Strassen+schoolbook decomposition."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        U[a00, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0; W[c11, col] += 1.0
        col += 1
        U[a10, col] += 1.0; U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0; W[c11, col] -= 1.0
        col += 1
        U[a00, col] += 1.0
        V[b01, col] += 1.0; V[b11, col] -= 1.0
        W[c01, col] += 1.0; W[c11, col] += 1.0
        col += 1
        U[a11, col] += 1.0
        V[b10, col] += 1.0; V[b00, col] -= 1.0
        W[c00, col] += 1.0; W[c10, col] += 1.0
        col += 1
        U[a00, col] += 1.0; U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0; W[c01, col] += 1.0
        col += 1
        U[a10, col] += 1.0; U[a00, col] -= 1.0
        V[b00, col] += 1.0; V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        U[a01, col] += 1.0; U[a11, col] -= 1.0
        V[b10, col] += 1.0; V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    for k in range(p):
        b2k = 2 * p + k
        U[a02, col] += 1.0; V[b2k, col] += 1.0; W[2 * k, col] += 1.0
        col += 1
        U[a12, col] += 1.0; V[b2k, col] += 1.0; W[2 * k + 1, col] += 1.0
        col += 1
    return U, V, W


def _build_rank_55_exact():
    """Build rank-55 decomposition with Fraction entries (for exact algebra)."""
    n, m, p = 2, 3, 10
    rank = 55
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    for k in range(p):
        b2k = 2 * p + k
        U[a02][col] += 1; V[b2k][col] += 1; W[2 * k][col] += 1
        col += 1
        U[a12][col] += 1; V[b2k][col] += 1; W[2 * k + 1][col] += 1
        col += 1
    return U, V, W


def _build_target_tensor_exact(n, m, p):
    """Return the target multiplication tensor as a dict {(i,j,k): 1} in flat form."""
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i * m + j, j * p + k, k * n + i)] = Fraction(1)
    return target


def _tensor_to_vector(target, n, m, p):
    """Flatten target tensor into a vector of Fractions of length n*m*p*n."""
    N = n * m * p * n
    vec = [Fraction(0)] * N
    for (a, b, c), val in target.items():
        idx = (a * (m * p) + b) * (p * n) + c
        vec[idx] = val
    return vec


def _decomposition_to_vector(U, V, W, n, m, p):
    """Flatten the decomposition T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t] into a vector."""
    N = n * m * p * n
    rank = len(U[0])
    vec = [Fraction(0)] * N
    for t in range(rank):
        # Collect non-zero entries in U[:,t], V[:,t], W[:,t]
        u_nz = [(a, U[a][t]) for a in range(n * m) if U[a][t] != 0]
        v_nz = [(b, V[b][t]) for b in range(m * p) if V[b][t] != 0]
        w_nz = [(c, W[c][t]) for c in range(p * n) if W[c][t] != 0]
        for a, ua in u_nz:
            for b, vb in v_nz:
                for c, wc in w_nz:
                    idx = (a * (m * p) + b) * (p * n) + c
                    vec[idx] += ua * vb * wc
    return vec


def _lll_reduce(rows):
    """
    Simple LLL-style reduction on a list of integer vectors.
    Returns a reduced basis (subset/recombination of rows with smaller norms).
    """
    if not rows:
        return []
    # Start with the input as a basis
    basis = [list(r) for r in rows]
    # Sort by norm
    basis.sort(key=lambda r: sum(x * x for x in r))
    # Greedy LLL-like: for each vector, reduce against earlier basis vectors
    reduced = []
    for v in basis:
        for i, b in enumerate(reduced):
            # Compute coefficient to subtract
            # Use the first non-zero coordinate of b
            nz = None
            for coord in range(len(b)):
                if b[coord] != 0:
                    nz = coord
                    break
            if nz is None:
                continue
            q = v[nz] // b[nz] if b[nz] > 0 else -((-v[nz]) // b[nz])
            if q != 0:
                for coord in range(len(v)):
                    v[coord] -= q * b[coord]
        # Check if v is in the span of reduced (all-zero after reduction)
        if all(x == 0 for x in v):
            continue
        # Insert v into reduced, maintaining order by norm
        norm_v = sum(x * x for x in v)
        inserted = False
        for i in range(len(reduced)):
            norm_b = sum(x * x for x in reduced[i])
            if norm_v < norm_b:
                reduced.insert(i, v)
                inserted = True
                break
        if not inserted:
            reduced.append(v)
    return reduced


def _check_linear_dependency_exact(start_time, deadline):
    """
    Check if the rank-55 decomposition has a linear dependency that allows
    a rank-54 exact solution. Uses exact Fraction arithmetic and LLL.
    Returns (True, U54, V54, W54) if a rank-54 decomposition is found, else (False, None, None, None).
    """
    n, m, p = 2, 3, 10
    U, V, W = _build_rank_55_exact()
    rank = 55
    
    target = _build_target_tensor_exact(n, m, p)
    target_vec = _tensor_to_vector(target, n, m, p)
    N = len(target_vec)
    
    # Build the "columns" of the decomposition: for each t, the vector
    # c_t[a,b,c] = U[a,t]*V[b,t]*W[c,t]
    # We want to check if {c_0, ..., c_54} are linearly dependent.
    # Build a matrix M where M[idx, t] = c_t[idx]
    # Then find the null space of M^T (i.e., vectors x such that sum_t x_t * c_t = 0).
    
    # Since N is large (6000) and rank is 55, we build M as a list of 55 column vectors.
    # Actually, let's build it as rows: for each idx where any c_t is non-zero, record the 55 values.
    
    # First, identify which indices are non-zero in any c_t
    nonzero_indices = set()
    for t in range(rank):
        u_nz = [a for a in range(n * m) if U[a][t] != 0]
        v_nz = [b for b in range(m * p) if V[b][t] != 0]
        w_nz = [c for c in range(p * n) if W[c][t] != 0]
        for a in u_nz:
            for b in v_nz:
                for c in w_nz:
                    idx = (a * (m * p) + b) * (p * n) + c
                    nonzero_indices.add(idx)
    
    # Build the matrix rows (only for nonzero indices)
    rows = []
    for idx in sorted(nonzero_indices):
        row = []
        for t in range(rank):
            # Find a,b,c from idx
            c = idx % (p * n)
            tmp = idx // (p * n)
            b = tmp % (m * p)
            a = tmp // (m * p)
            row.append(U[a][t] * V[b][t] * W[c][t])
        rows.append(row)
    
    num_rows = len(rows)
    
    # Now we need to find the null space of this num_rows x 55 matrix.
    # Use Gaussian elimination over Q.
    # Convert to a list of lists of Fractions
    mat = [list(r) for r in rows]
    
    # Gaussian elimination
    pivot_cols = []
    for col in range(rank):
        if time.time() > deadline:
            return (False, None, None, None)
        # Find pivot row
        pivot_row = None
        for r in range(len(pivot_cols), len(mat)):
            if mat[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        # Swap
        mat[len(pivot_cols)], mat[pivot_row] = mat[pivot_row], mat[len(pivot_cols)]
        pivot_cols.append(col)
        # Eliminate
        pivot_val = mat[len(pivot_cols) - 1][col]
        for r in range(len(mat)):
            if r == len(pivot_cols) - 1:
                continue
            factor = mat[r][col] / pivot_val
            if factor != 0:
                for c in range(col, rank):
                    mat[r][c] -= factor * mat[len(pivot_cols) - 1][c]
    
    # The null space has dimension rank - len(pivot_cols)
    nullity = rank - len(pivot_cols)
    
    if nullity == 0:
        return (False, None, None, None)
    
    # Find a basis for the null space
    # For each free column, construct a null vector
    free_cols = [c for c in range(rank) if c not in pivot_cols]
    null_basis = []
    for fc in free_cols:
        v = [Fraction(0)] * rank
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivot_cols):
            v[pc] = -mat[i][fc] / mat[i][pc]
        null_basis.append(v)
    
    # Try to find a null vector with small integer entries using LLL
    # Convert null basis to integer vectors by clearing denominators
    int_basis = []
    for v in null_basis:
        # Find LCM of denominators
        lcms = [1]
        for x in v:
            if x != 0:
                lcms.append(x.denominator)
        lcm = 1
        for d in lcms:
            g = 1
            a, b = max(lcm, d), min(lcm, d)
            while b:
                a, b = b, a % b
            lcm = lcm * d // g
        int_v = [int(x * lcm) for x in v]
        int_basis.append(int_v)
    
    # Apply LLL-like reduction to find a short vector
    reduced = _lll_reduce(int_basis)
    
    # Check each reduced vector: if it has a zero in some coordinate,
    # we can drop that term and get a rank-54 decomposition
    for vec in reduced:
        for t in range(rank):
            if vec[t] == 0:
                # This null vector has zero coefficient for term t.
                # But we need the null vector to be non-trivial.
                # If vec[t] == 0 and the vector is non-zero, we can't directly drop term t
                # unless we can express the remaining terms as a valid decomposition.
                # Actually, if sum_s vec[s] * c_s = 0 and vec[t] = 0, then
                # sum_{s != t} vec[s] * c_s = 0.
                # This doesn't directly give a rank-54 decomposition.
                # We need a different approach.
                pass
    
    # Alternative: check if any single term c_t is in the span of the others.
    # For each t, check if c_t is in span{c_s : s != t}.
    for t_drop in range(rank):
        if time.time() > deadline:
            return (False, None, None, None)
        # Check if c_t is in the span of the other 54 vectors
        # Build the matrix without column t and check if c_t is in its column space
        # This is equivalent to checking if the 55x55 matrix with column t removed
        # has rank 55 (impossible) or if c_t is in the span.
        
        # Simpler: check if the 55 vectors (excluding t) span c_t.
        # Build a system: for each row r, sum_{s != t} x_s * M[r,s] = M[r,t]
        # This is a linear system with 54 unknowns and num_rows equations.
        # Use Gaussian elimination.
        
        # Build augmented matrix [M_without_t | c_t]
        aug = []
        for r in range(num_rows):
            row = [mat[r][s] for s in range(rank) if s != t_drop]
            row.append(mat[r][t_drop])
            aug.append(row)
        
        # Gaussian elimination
        n_unknowns = rank - 1
        pivot_cols_aug = []
        for col in range(n_unknowns):
            if time.time() > deadline:
                return (False, None, None, None)
            pivot_row = None
            for r in range(len(pivot_cols_aug), len(aug)):
                if aug[r][col] != 0:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue
            aug[len(pivot_cols_aug)], aug[pivot_row] = aug[pivot_row], aug[len(pivot_cols_aug)]
            pivot_cols_aug.append(col)
            pivot_val = aug[len(pivot_cols_aug) - 1][col]
            for r in range(len(aug)):
                if r == len(pivot_cols_aug) - 1:
                    continue
                factor = aug[r][col] / pivot_val
                if factor != 0:
                    for c in range(col, n_unknowns + 1):
                        aug[r][c] -= factor * aug[len(pivot_cols_aug) - 1][c]
        
        # Check consistency: for rows where all coefficients are zero, the RHS must be zero
        consistent = True
        for r in range(len(aug)):
            all_zero = all(aug[r][c] == 0 for c in range(n_unknowns))
            if all_zero and aug[r][n_unknowns] != 0:
                consistent = False
                break
        
        if consistent:
            # c_t is in the span of the others.
            # We can drop term t and get a rank-54 decomposition.
            # But we need to verify that the remaining 54 terms still span the target.
            # Since c_t is in the span of the others, and the full set spans the target,
            # the remaining 54 also span the target.
            # Construct the rank-54 decomposition by dropping term t.
            U54 = [[U[a][s] for s in range(rank) if s != t_drop] for a in range(n * m)]
            V54 = [[V[b][s] for s in range(rank) if s != t_drop] for b in range(m * p)]
            W54 = [[W[c][s] for s in range(rank) if s != t_drop] for c in range(p * n)]
            return (True, U54, V54, W54)
    
    return (False, None, None, None)
    """
    Constructs the rank-55 decomposition.
    Split m=3 into m1=2 (Strassen) and m2=1 (Schoolbook).
    Split p=10 into 5 blocks of 2 for the Strassen part.
    """
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Indices for A (2x3)
    # A[i,j] -> U[i*m + j]
    # A00 -> 0, A01 -> 1, A02 -> 2
    # A10 -> 3, A11 -> 4, A12 -> 5
    a00, a01, a02 = 0, 1, 2
    a10, a11, a12 = 3, 4, 5
    
    # Process 5 blocks of 2 columns for the Strassen part (A_L * B_L)
    # A_L is 2x2 (cols 0,1), B_L is 2x10 (rows 0,1)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Indices for B (3x10)
        # B[j,k] -> V[j*p + k]
        # B00 -> V[0], B01 -> V[1]
        # B10 -> V[10], B11 -> V[11]
        b00, b01 = k0, k1
        b10, b11 = 10 + k0, 10 + k1
        
        # Indices for C (2x10 output)
        # C[i,k] -> W[k*n + i]
        # C00 -> W[0], C01 -> W[2]
        # C10 -> W[1], C11 -> W[3]
        c00, c01 = 2 * k0, 2 * k1
        c10, c11 = 2 * k0 + 1, 2 * k1 + 1
        
        # Strassen's algorithm for 2x2 matrix multiplication
        # M1 = (A00 + A11)(B00 + B11)
        U[a00, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M2 = (A10 + A11)B00
        U[a10, col] += 1.0
        U[a11, col] += 1.0
        V[b00, col] += 1.0
        W[c10, col] += 1.0
        W[c11, col] -= 1.0
        col += 1
        
        # M3 = A00(B01 - B11)
        U[a00, col] += 1.0
        V[b01, col] += 1.0
        V[b11, col] -= 1.0
        W[c01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M4 = A11(B10 - B00)
        U[a11, col] += 1.0
        V[b10, col] += 1.0
        V[b00, col] -= 1.0
        W[c00, col] += 1.0
        W[c10, col] += 1.0
        col += 1
        
        # M5 = (A00 + A01)B11
        U[a00, col] += 1.0
        U[a01, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] -= 1.0
        W[c01, col] += 1.0
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[a10, col] += 1.0
        U[a00, col] -= 1.0
        V[b00, col] += 1.0
        V[b01, col] += 1.0
        W[c11, col] += 1.0
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[a01, col] += 1.0
        U[a11, col] -= 1.0
        V[b10, col] += 1.0
        V[b11, col] += 1.0
        W[c00, col] += 1.0
        col += 1
    
    # Schoolbook part for A_R * B_R (2x1 * 1x10)
    # A_R is 2x1: A02 (idx 2), A12 (idx 5)
    # B_R is 1x10: B20..B29 (idx 20..29)
    # C contributions: C00..C09 (idx 0,2..18), C10..C19 (idx 1,3..19)
    
    for k in range(p):
        b2k = 2 * p + k  # Index for B[2, k]
        
        # Term 1: A02 * B2k -> C0k
        # C0k is at W[k*n + 0] = W[2*k]
        U[a02, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k, col] += 1.0
        col += 1
        
        # Term 2: A12 * B2k -> C1k
        # C1k is at W[k*n + 1] = W[2*k + 1]
        U[a12, col] += 1.0
        V[b2k, col] += 1.0
        W[2 * k + 1, col] += 1.0
        col += 1
        
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = 2, 3, 10
    
    # Build the rank-55 decomposition
    U, V, W = _build_rank_55()
    
    # Verify
    if time.time() - start_time < DEADLINE:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback (should not happen)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
