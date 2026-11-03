"""Structured low-rank bilinear scheme + exact greedy rank reduction for 3x3 @ 3x6."""

import fractions
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 6]


def _row_vec(n, idx, val):
    v = [Fraction(0)] * n
    v[idx] = Fraction(val)
    return v


def _structured_base():
    """Return (U,V,W) for a known valid bilinear scheme of rank R=27.
    U: (n*m)=9 rows, V: (m*p)=18 rows, W: (p*n)=18 rows, each R cols."""
    n, m, p = DIMENSIONS
    # Strassen-style 7-term scheme for 3x3*3x3 (A,B,C are 3x3 with C=A*B)
    base = [
        # (U-row, U-col, V-row, V-col, W-row, W-col)  rows indexed by (i,j) / (j,k) / (k,i)
        ((0, 0), 1, (0, 0), 1, (0, 0), 1),
        ((1, 0), 1, (0, 1), 1, (1, 0), 1),
        ((1, 1), -1, (0, 0), 1, (1, 1), -1),
        ((0, 0), -1, (1, 0), 1, (0, 0), -1),
        ((0, 1), 1, (1, 1), 1, (0, 1), 1),
        ((2, 0), 1, (0, 2), 1, (2, 0), 1),
        ((2, 1), -1, (0, 0), 1, (2, 1), -1),
        ((0, 0), -1, (2, 0), 1, (0, 0), -1),
        ((0, 2), 1, (2, 2), 1, (0, 2), 1),
        ((1, 0), -1, (2, 1), 1, (1, 0), -1),
        ((1, 1), 1, (2, 1), 1, (1, 1), 1),
        ((2, 0), -1, (2, 2), 1, (2, 0), -1),
        ((2, 2), 1, (2, 0), 1, (2, 2), 1),
    ]
    # Expand: each base term (for k0 in 0..2) -> term for output column k = k0 + 3*term_idx
    terms = []  # list of (U_idx, U_val, V_idx, V_val, W_idx, W_val)
    for t in range(len(base)):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = base[t]
        for k0 in range(p):
            k = k0 + 3 * t
            terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val, k))

    R = len(terms)
    U = [[Fraction(0)] * R for _ in range(n * m)]
    V = [[Fraction(0)] * R for _ in range(m * p)]
    W = [[Fraction(0)] * R for _ in range(p * n)]
    for c, (u_idx, u_val, v_idx, v_val, w_idx, w_val, k) in enumerate(terms):
        i, j = u_idx
        a, b = v_idx
        cc, d = w_idx
        U[i * m + j][c] = Fraction(u_val)
        V[a * p + b][c] = Fraction(v_val)
        W[cc * n + d][c] = Fraction(w_val)
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


def _greedy_reduce(U, V, W):
    """Try to remove a column that is a rational linear combination of others."""
    n, m, p = DIMENSIONS
    R = len(W[0]) if W else 0
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    best = (U, V, W, R)
    
    # Optimization: Pre-compute the tensor entries for the current basis
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We want to see if T can be represented with R-1 terms.
    
    for c0 in range(R):
        cols = list(range(R))
        cols.remove(c0)
        # Build system: n*m*p rows x (R-1) cols
        A = []
        b = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for c in cols:
                        row.append(U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])
                    A.append(row)
                    b.append(U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0])
        
        # Gaussian elimination to solve A * a = b exactly
        nr = len(A)
        nc = len(cols)
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
        
        consistent = True
        for rr in range(nr):
            allz = all(M[rr][cc2] == 0 for cc2 in range(nc))
            if allz and M[rr][nc] != 0:
                consistent = False
                break
        
        if not consistent:
            continue
        
        sol = [Fraction(0)] * nc
        for col in pivots:
            sol[col] = M[col][nc]
            
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for idx_c, c in enumerate(cols):
                        s += sol[idx_c] * U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != U[i * m + j][c0] * V[j * p + k][c0] * W[k * n + i][c0]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            continue
        
        newU = [row[:c0] + row[c0 + 1:] for row in U]
        newV = [row[:c0] + row[c0 + 1:] for row in V]
        newW = [row[:c0] + row[c0 + 1:] for row in W]
        
        if _validate(newU, newV, newW):
            R2 = R - 1
            if R2 < best[3]:
                best = (newU, newV, newW, R2)
                U, V, W, R = newU, newV, newW, R2
    return best


def _random_basis_change(U, V, W, R):
    """Apply a random invertible linear transformation to the columns of U, V, W."""
    # Generate random matrix P of size R x R with Fraction(1) on diagonal and small random off-diagonal?
    # To keep it simple and exact, we can just swap columns or add one column to another.
    # Let's do a few random column additions.
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    for _ in range(5):
        c1 = random.randint(0, R - 1)
        c2 = random.randint(0, R - 1)
        if c1 == c2:
            continue
        # U[:, c1] += U[:, c2]
        for r in range(len(U)):
            U[r][c1] += U[r][c2]
        # V[:, c1] += V[:, c2]
        for r in range(len(V)):
            V[r][c1] += V[r][c2]
        # W[:, c1] += W[:, c2]
        for r in range(len(W)):
            W[r][c1] += W[r][c2]
    return U, V, W


def construct():
    U, V, W, R = _structured_base()
    if not _validate(U, V, W):
        n, m, p = DIMENSIONS
        rs = n * m * p
        U = [[0] * rs for _ in range(n * m)]
        V = [[0] * rs for _ in range(m * p)]
        W = [[0] * rs for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce
    U, V, W, R = _greedy_reduce(U, V, W)
    
    # If no reduction, try random basis changes and reduce again
    best_U, best_V, best_W, best_R = U, V, W, R
    for _ in range(10):
        U_rand, V_rand, W_rand = _random_basis_change(U, V, W, R)
        U_red, V_red, W_red, R_red = _greedy_reduce(U_rand, V_rand, W_rand)
        if R_red < best_R:
            best_U, best_V, best_W, best_R = U_red, V_red, W_red, R_red
    
    U, V, W, R = best_U, best_V, best_W, best_R
    
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
