"""Exact rank-minimal tensor search for 3x3x8 matmul.

Strategy:
- Build a rank-68 decomposition using 4 disjoint 2x2 Strassen blocks.
- Apply structural merges and targeted cancellations to reduce rank further.
- Verify every candidate with exact integer arithmetic.
- Return the lowest-rank exact decomposition found before the deadline.
"""
import time
import itertools
from fractions import Fraction

_START = time.time()
DEADLINE = 140.0  # safety margin under 150s kill


def _time_left():
    return DEADLINE - (time.time() - _START)


def _check(U, V, W, n, m, p):
    """Exact verification: U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c] sums to
    1 iff i==k else 0."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1 if i == k else 0
                s = 0
                ui = i * m + j
                vj = j * p
                wk = k * n + i
                for c in range(rank):
                    u = U[ui][c]
                    if u:
                        v = V[vj + k][c]
                        if v:
                            w = W[wk][c]
                            if w:
                                s += u * v * w
                if s != target:
                    return False
    return True


def _schoolbook(n, m, p):
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _merge_identical(U, V, W, n, m, p):
    """Merge rank-1 terms whose (u,v) pattern is identical: their w patterns add."""
    rank = len(U[0])
    if rank == 0:
        return U, V, W
    groups = {}
    for c in range(rank):
        up = tuple(U[i * m + j][c] for i in range(n) for j in range(m))
        vp = tuple(V[j * p + k][c] for j in range(m) for k in range(p))
        key = (up, vp)
        if key not in groups:
            groups[key] = []
        groups[key].append(c)
    
    newU_cols = []
    newV_cols = []
    newW_cols = []
    for (up, vp), cols in groups.items():
        if len(cols) == 1:
            c = cols[0]
            newU_cols.append([U[i * m + j][c] for i in range(n) for j in range(m)])
            newV_cols.append([V[j * p + k][c] for j in range(m) for k in range(p)])
            newW_cols.append([W[k * n + i][c] for k in range(p) for i in range(n)])
        else:
            newU_cols.append(list(up))
            newV_cols.append(list(vp))
            nw = [0] * (p * n)
            for c in cols:
                for idx in range(p * n):
                    nw[idx] += W[idx][c]
            newW_cols.append(nw)
    
    nr = len(newU_cols)
    U2 = [[newU_cols[c][i] for c in range(nr)] for i in range(n * m)]
    V2 = [[newV_cols[c][j] for c in range(nr)] for j in range(m * p)]
    W2 = [[newW_cols[c][k] for c in range(nr)] for k in range(p * n)]
    return U2, V2, W2


def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if (i0+di) < n and (j0+dj) < m and (k0+dk) < p:
                        covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    ii = i0 + di
                    jj = j0 + dj
                    if ii < n and jj < m:
                        U[ii*m + jj][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    jj = j0 + dj
                    kk = k0 + dk
                    if jj < m and kk < p:
                        V[jj*p + kk][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    kk = k0 + dk
                    ii = i0 + di
                    if kk < p and ii < n:
                        W[kk*n + ii][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W


def _try_remove_term(U, V, W, n, m, p, c_remove):
    """Try to remove term c_remove by expressing it as linear combo of others.
    Returns (newU, newV, newW) or None."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    rem = [c for c in range(rank) if c != c_remove]
    
    def tensor_vec(c2):
        vec = [0] * (n * m * p)
        for i in range(n):
            for j in range(m):
                u = U[i * m + j][c2]
                if u:
                    for k in range(p):
                        v = V[j * p + k][c2]
                        if v:
                            w = W[k * n + i][c2]
                            if w:
                                vec[i * m * p + j * p + k] = u * v * w
        return vec
    
    target = tensor_vec(c_remove)
    # Check if target is zero vector (term contributes nothing)
    if all(x == 0 for x in target):
        newU = [row[:c_remove] + row[c_remove+1:] for row in U]
        newV = [row[:c_remove] + row[c_remove+1:] for row in V]
        newW = [row[:c_remove] + row[c_remove+1:] for row in W]
        return newU, newV, newW
    
    # Try to find small integer coefficients
    # Use a simple approach: check if target is a linear combination with small coefficients
    # For efficiency, only try if the system is likely to have a solution
    ncols = len(rem)
    
    # Build the matrix and solve
    A = [tensor_vec(r) for r in rem]
    mrows = len(target)
    
    # Gaussian elimination over rationals
    M = [[Fraction(A[r][row]) for r in range(ncols)] + [Fraction(target[row])] for row in range(mrows)]
    
    r = 0
    pivot_cols = []
    for col in range(ncols):
        if _time_left() < 2.0:
            return None
        pivot = None
        for rr in range(r, mrows):
            if M[rr][col] != 0:
                pivot = rr
                break
        if pivot is None:
            continue
        M[r], M[pivot] = M[pivot], M[r]
        pv = M[r][col]
        for cc in range(col, ncols + 1):
            M[r][cc] /= pv
        for rr in range(mrows):
            if rr == r:
                continue
            f = M[rr][col]
            if f != 0:
                for cc in range(col, ncols + 1):
                    M[rr][cc] -= f * M[r][cc]
        r += 1
        pivot_cols.append(col)
    
    # Check consistency
    ok = True
    for rr in range(r, mrows):
        if M[rr][ncols] != 0:
            ok = False
            break
    if not ok:
        return None
    
    # Extract solution
    a = [0] * ncols
    for rr in range(r):
        pcol = pivot_cols[rr]
        a[pcol] = M[rr][ncols]
    
    # Check all are integers
    if not all(x.denominator == 1 for x in a):
        return None
    
    # Verify the solution
    # sum a_r * tensor_vec(rem[r]) should equal target
    verify = [0] * (n * m * p)
    for idx, r in enumerate(rem):
        if a[idx] != 0:
            tv = tensor_vec(r)
            for pos in range(len(verify)):
                verify[pos] += a[idx] * tv[pos]
    
    if verify != target:
        return None
    
    # Remove the term
    newU = [row[:c_remove] + row[c_remove+1:] for row in U]
    newV = [row[:c_remove] + row[c_remove+1:] for row in V]
    newW = [row[:c_remove] + row[c_remove+1:] for row in W]
    
    if _check(newU, newV, newW, n, m, p):
        return newU, newV, newW
    
    return None


def _greedy_eliminate(U, V, W, n, m, p, max_terms=3):
    """Try to remove up to max_terms using elimination."""
    improved = True
    count = 0
    while improved and count < max_terms and _time_left() > 5.0:
        improved = False
        count += 1
        rank = len(U[0])
        # Try removing each term, prefer those with sparse support
        supports = []
        for c in range(rank):
            sup = sum(1 for row in U for val in row if val != 0)
            supports.append((sup, c))
        supports.sort()
        for _, c in supports:
            if _time_left() < 3.0:
                break
            cand = _try_remove_term(U, V, W, n, m, p, c)
            if cand is not None:
                U, V, W = cand
                improved = True
                break
    return U, V, W


def construct():
    n, m, p = 3, 3, 8
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Define 4 non-overlapping Strassen blocks on the 3x3x8 tensor
    # i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
    # Use blocks: (0,0,0), (0,0,2), (0,0,4), (0,0,6)
    # These cover i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5,6,7}
    # Wait, each block covers 2x2x2 = 8 elements
    # 4 blocks cover 32 elements, but we only have 3*3*8=72
    # The blocks should be non-overlapping
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={0,1}, k={2,3}
    # Block 3: i={0,1}, j={0,1}, k={4,5}
    # Block 4: i={0,1}, j={0,1}, k={6,7}
    # This covers i in {0,1}, j in {0,1}, all k. That's 2*2*8=32 elements.
    # Remaining: i=2 (all j,k) = 3*8=24, and i in {0,1}, j=2, all k = 2*1*8=16
    # Total remaining: 24+16=40. 32+40=72. Good.
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6)
    ]
    
    if _time_left() > 10:
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            U, V, W = _merge_identical(U, V, W, n, m, p)
            if _check(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try to reduce further with greedy elimination
    if _time_left() > 15:
        try:
            U, V, W = best_result["U"], best_result["V"], best_result["W"]
            U, V, W = _merge_identical(U, V, W, n, m, p)
            U, V, W = _greedy_eliminate(U, V, W, n, m, p, max_terms=5)
            U, V, W = _merge_identical(U, V, W, n, m, p)
            if _check(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try additional Strassen-like reductions on the remaining structure
    if _time_left() > 20:
        try:
            U, V, W = best_result["U"], best_result["V"], best_result["W"]
            # Try more aggressive merges
            for _ in range(3):
                U, V, W = _merge_identical(U, V, W, n, m, p)
                U, V, W = _greedy_eliminate(U, V, W, n, m, p, max_terms=3)
            U, V, W = _merge_identical(U, V, W, n, m, p)
            if _check(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Final verification
    if not _check(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    return best_result
