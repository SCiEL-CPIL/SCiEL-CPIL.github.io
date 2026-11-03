"""Schoolbook base + aggressive iterative greedy rank reduction for 3x3 @ 3x6."""

import fractions
from fractions import Fraction

DIMENSIONS = [3, 3, 6]


def _schoolbook_base():
    """Return (U,V,W) for schoolbook bilinear scheme of rank R=n*m*p=54.
    U: (n*m)=9 rows, V: (m*p)=18 rows, W: (p*n)=18 rows, each R cols."""
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W, R


def _validate(U, V, W):
    """Check U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed over c == 1 for every (i,j,k)."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _try_remove_column(U, V, W, c0):
    """Try to remove column c0 by finding coefficients for remaining columns.
    Returns (newU, newV, newW) if successful, else None."""
    n, m, p = DIMENSIONS
    R = len(W[0])
    cols = list(range(R))
    cols.remove(c0)
    nc = len(cols)
    nr = n * m * p
    # Build system A * a = b
    # A[r, c_idx] = U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] for the c-th remaining column
    # b[r] = U[i*m+j][c0]*V[j*p+k][c0]*W[k*n+i][c0]
    # Map (i,j,k) to row index
    def row_idx(i, j, k):
        return (i * m + j) * p + k
    
    A = [[Fraction(0)] * nc for _ in range(nr)]
    b = [Fraction(0)] * nr
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = row_idx(i, j, k)
                for idx_c, c in enumerate(cols):
                    A[r][idx_c] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                b[r] = U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0]
    
    # Gaussian elimination
    M = [A[r][:] + [b[r]] for r in range(nr)]
    pivots = []
    for col in range(nc):
        piv_row = None
        for rr in range(col, nr):
            if M[rr][col] != 0:
                piv_row = rr
                break
        if piv_row is None:
            continue
        if piv_row != col:
            M[col], M[piv_row] = M[piv_row], M[col]
        pivots.append(col)
        piv_val = M[col][col]
        inv = Fraction(1) / piv_val
        for cc2 in range(col, nc + 1):
            M[col][cc2] = M[col][cc2] * inv
        for rr in range(nr):
            if rr == col:
                continue
            f = M[rr][col]
            if f == 0:
                continue
            for cc2 in range(col, nc + 1):
                M[rr][cc2] = M[rr][cc2] - f * M[col][cc2]
    
    # Check consistency
    for rr in range(nr):
        allz = all(M[rr][cc2] == 0 for cc2 in range(nc))
        if allz and M[rr][nc] != 0:
            return None
    
    # Get particular solution
    sol = [Fraction(0)] * nc
    for col in pivots:
        sol[col] = M[col][nc]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for idx_c, c in enumerate(cols):
                    s += sol[idx_c] * U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0]:
                    return None
    
    newU = [row[:c0] + row[c0 + 1:] for row in U]
    newV = [row[:c0] + row[c0 + 1:] for row in V]
    newW = [row[:c0] + row[c0 + 1:] for row in W]
    return newU, newV, newW


def _greedy_reduce(U, V, W):
    """Iteratively try to remove columns."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    improved = True
    while improved:
        improved = False
        R = len(W[0])
        for c0 in range(R):
            result = _try_remove_column(U, V, W, c0)
            if result is not None:
                newU, newV, newW = result
                if _validate(newU, newV, newW):
                    U, V, W = newU, newV, newW
                    improved = True
                    break
    return U, V, W, len(W[0]) if W else 0


def _random_perturbation(U, V, W, seed=42):
    """Apply random invertible transformations to U, V, W to enable further reduction.
    U' = U * P, V' = V * P^{-1}, W' = W (for some invertible matrix P)
    This preserves the decomposition but changes which columns are removable."""
    import random
    rng = random.Random(seed)
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    if R <= 1:
        return U, V, W
    
    # Generate random invertible matrix P (R x R) over Q
    # Use random upper triangular with 1s on diagonal
    P = [[Fraction(0)] * R for _ in range(R)]
    for i in range(R):
        P[i][i] = Fraction(1)
        for j in range(i + 1, R):
            P[i][j] = Fraction(rng.randint(-2, 2))
    
    # Compute P^{-1}
    # Augmented matrix [P | I]
    aug = [[P[i][j] for j in range(R)] + ([Fraction(1) if i == j else Fraction(0) for j in range(R)]) for i in range(R)]
    for col in range(R):
        piv_row = None
        for rr in range(col, R):
            if aug[rr][col] != 0:
                piv_row = rr
                break
        if piv_row is None:
            return U, V, W  # P not invertible, skip
        if piv_row != col:
            aug[col], aug[piv_row] = aug[piv_row], aug[col]
        piv_val = aug[col][col]
        inv = Fraction(1) / piv_val
        for cc2 in range(2 * R):
            aug[col][cc2] = aug[col][cc2] * inv
        for rr in range(R):
            if rr == col:
                continue
            f = aug[rr][col]
            if f == 0:
                continue
            for cc2 in range(2 * R):
                aug[rr][cc2] = aug[rr][cc2] - f * aug[col][cc2]
    
    P_inv = [aug[i][R:] for i in range(R)]
    
    # U' = U * P: U'[i][c'] = sum_c U[i][c] * P[c][c']
    newU = [[Fraction(0)] * R for _ in range(len(U))]
    for i in range(len(U)):
        for c in range(R):
            if U[i][c] == 0:
                continue
            for c2 in range(c, R):
                if P[c][c2] != 0:
                    newU[i][c2] += U[i][c] * P[c][c2]
    
    # V' = V * P_inv: V'[i][c'] = sum_c V[i][c] * P_inv[c][c']
    newV = [[Fraction(0)] * R for _ in range(len(V))]
    for i in range(len(V)):
        for c in range(R):
            if V[i][c] == 0:
                continue
            for c2 in range(c, R):
                if P_inv[c][c2] != 0:
                    newV[i][c2] += V[i][c] * P_inv[c][c2]
    
    newW = [list(row) for row in W]
    
    return newU, newV, newW


def construct():
    U, V, W, R = _schoolbook_base()
    
    # Try multiple random perturbations to find a better reducible form
    best_rank = R
    best_U, best_V, best_W = U, V, W
    
    for seed in range(20):
        if seed == 0:
            U_try, V_try, W_try = U, V, W
        else:
            U_try, V_try, W_try = _random_perturbation(U, V, W, seed)
        
        U_red, V_red, W_red, R_red = _greedy_reduce(U_try, V_try, W_try)
        
        if R_red < best_rank:
            best_rank = R_red
            best_U, best_V, best_W = U_red, V_red, W_red
    
    U, V, W = best_U, best_V, best_W
    
    # Convert Fractions to int or float for output
    def _conv(row):
        out = []
        for x in row:
            if x.denominator == 1:
                out.append(int(x.numerator))
            else:
                out.append(float(x))
        return out
    U = [_conv(r) for r in U]
    V = [_conv(r) for r in V]
    W = [_conv(r) for r in W]
    return {"U": U, "V": V, "W": W}
