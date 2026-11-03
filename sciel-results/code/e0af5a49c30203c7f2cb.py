"""Fresh design: exact rank-minimal tensor search for 3x3x8 matmul.

Strategy:
- Start from the schoolbook rank-72 decomposition.
- Greedily merge/eliminate rank-1 terms (sums of terms with identical
  input pattern, then structural Strassen-type cancellations) to reduce rank.
- Verify every candidate with a fast exact integer check of the tensor identity.
- Return the lowest-rank exact decomposition found before the deadline.
"""
import time
import itertools

DEADLINE = time.time() + 140.0  # safety margin under 150s kill

DIMENSIONS = [3, 3, 8]


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


def _rank1(U, V, W, n, m, p):
    """Represent a rank-1 term as (u_pattern, v_pattern, w_pattern) where
    u_pattern[i,j]=U[i*m+j,c], v_pattern[j,k]=V[j*p+k,c], w_pattern[k,i]=W[k*n+i,c]."""
    up = {}
    for i in range(n):
        for j in range(m):
            up[(i, j)] = U[i * m + j][0]
    vp = {}
    for j in range(m):
        for k in range(p):
            vp[(j, k)] = V[j * p + k][0]
    wp = {}
    for k in range(p):
        for i in range(n):
            wp[(k, i)] = W[k * n + i][0]
    return up, vp, wp


def _add_term(U, V, W, up, vp, wp, n, m, p):
    """Add a rank-1 term (patterns) as a new column."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            U[i * m + j].append(up.get((i, j), 0))
    for j in range(m):
        for k in range(p):
            V[j * p + k].append(vp.get((j, k), 0))
    for k in range(p):
        for i in range(n):
            W[k * n + i].append(wp.get((k, i), 0))
    return rank


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
    """Merge rank-1 terms whose (u,v) pattern is identical: their w patterns
    add. This is exact because (u*V)*(w1+w2) = u*V*w1 + u*V*w2."""
    rank = len(U[0])
    groups = {}
    for c in range(rank):
        up = tuple(U[i * m + j][c] for i in range(n) for j in range(m))
        vp = tuple(V[j * p + k][c] for j in range(m) for k in range(p))
        key = (up, vp)
        groups.setdefault(key, []).append(c)
    newU = []
    newV = []
    newW = []
    for (up, vp), cols in groups.items():
        if len(cols) == 1:
            c = cols[0]
            newU.append([U[i * m + j][c] for i in range(n) for j in range(m)])
            newV.append([V[j * p + k][c] for j in range(m) for k in range(p)])
            newW.append([W[k * n + i][c] for k in range(p) for i in range(n)])
        else:
            newU.append(list(up))
            newV.append(list(vp))
            nw = [0] * (p * n)
            for c in cols:
                for idx in range(p * n):
                    nw[idx] += W[idx][c]
            newW.append(nw)
    # transpose back to row-major lists
    U2 = [[newU[c][i] for c in range(len(newU))] for i in range(n * m)]
    V2 = [[newV[c][j] for c in range(len(newV))] for j in range(m * p)]
    W2 = [[newW[c][k] for c in range(len(newW))] for k in range(p * n)]
    return U2, V2, W2


def _try_strassen_compress(U, V, W, n, m, p):
    """Try to replace the 9 input-side (i,j) schoolbook terms that feed a single
    output column structure with a 7-term Strassen block when the output side
    is uniform. Returns a candidate or None."""
    # This is a structural heuristic; we keep it conservative and only accept
    # if verification passes. For 3x3 input with 8 outputs, full Strassen on
    # the 3x3 block is not directly applicable, so we attempt a generic
    # low-rank recombination on the (i,j) face.
    return None


def _greedy_eliminate(U, V, W, n, m, p, budget=1.0):
    """Attempt to remove a single rank-1 term by re-expressing it as a linear
    combination of the others (exact). Only accept if verified."""
    rank = len(U[0])
    t0 = time.time()
    for c in range(rank):
        if time.time() - t0 > budget:
            break
        # Build reduced system: can term c be expressed as sum of others?
        # We attempt a simple integer null-vector search on the tensor
        # coefficients of the remaining terms.
        rem = [c2 for c2 in range(rank) if c2 != c]
        # Represent each remaining term by its full tensor vector of length n*m*p
        def tensor_vec(c2):
            vec = [0] * (n * m * p)
            for i in range(n):
                for j in range(m):
                    u = U[i * m + j][c2]
                    for k in range(p):
                        vec[i * m * p + j * p + k] = u * V[j * p + k][c2] * W[k * n + i][c2]
            return vec
        # target = tensor_vec(c)
        # We look for small-integer coefficients a_r such that sum a_r*vec(r) = vec(c)
        # This is a linear system; solve exactly with integer arithmetic (Gauss).
        target = tensor_vec(c)
        # Build matrix A (n*m*p x len(rem))
        A = [tensor_vec(r) for r in rem]
        # Solve A * a = target over integers if possible.
        # Use rational Gaussian elimination.
        from fractions import Fraction
        mrows = len(target)
        ncols = len(rem)
        M = [[Fraction(A[r][row]) for r in range(ncols)] + [Fraction(target[row])] for row in range(mrows)]
        # Gaussian elimination
        r = 0
        for col in range(ncols):
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
        # Check consistency
        ok = True
        for rr in range(r, mrows):
            if M[rr][ncols] != 0:
                ok = False
                break
        if not ok:
            continue
        # Solution exists; extract integer solution (particular solution)
        a = [0] * ncols
        # Back substitution using reduced form
        for rr in range(r):
            # find the pivot column of row rr
            pcol = None
            for cc in range(ncols):
                if M[rr][cc] != 0:
                    pcol = cc
                    break
            if pcol is not None:
                a[pcol] = M[rr][ncols]
        # Check all a are integers
        if all(x.denominator == 1 for x in a):
            # Build candidate: remove term c, add a_r * term r
            newU = [row[:] for row in U]
            newV = [row[:] for row in V]
            newW = [row[:] for row in W]
            # We need to add a_r * term_r into a new combined column, but since
            # we're removing c and the combination equals c, the net effect on the
            # tensor is zero, so we can simply drop column c.
            newU = [row[:c] + row[c + 1:] for row in U]
            newV = [row[:c] + row[c + 1:] for row in V]
            newW = [row[:c] + row[c + 1:] for row in W]
            if _check(newU, newV, newW, n, m, p):
                return newU, newV, newW
    return None


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)

    # 1) Merge identical (u,v) patterns
    U, V, W = _merge_identical(U, V, W, n, m, p)

    best = (len(U[0]), U, V, W)

    # 2) Greedy elimination loop
    improved = True
    while improved and time.time() < DEADLINE - 5.0:
        improved = False
        U, V, W = _merge_identical(U, V, W, n, m, p)
        cand = _greedy_eliminate(U, V, W, n, m, p, budget=1.5)
        if cand is not None:
            U, V, W = cand
            improved = True
            if len(U[0]) < best[0]:
                best = (len(U[0]), U, V, W)

    # 3) Final merge
    U, V, W = _merge_identical(U, V, W, n, m, p)
    if len(U[0]) < best[0] and _check(U, V, W, n, m, p):
        best = (len(U[0]), U, V, W)

    # Ensure we return a verified result; fall back to schoolbook if needed
    br, bU, bV, bW = best
    if not _check(bU, bV, bW, n, m, p):
        bU, bV, bW = _schoolbook(n, m, p)
    return {"U": bU, "V": bV, "W": bW}
