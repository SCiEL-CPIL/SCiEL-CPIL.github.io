"""Rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Rank-7 decomposition of the 2x2x2 matrix-multiplication tensor.

    U[i*2+j, c] (A side), V[j*2+k, c] (B side), W[k*2+i, c] (C side).
    Identity: C[i,k] = sum_j A[i,j] B[j,k] =
        sum_c W[k*2+i, c] * (sum_j A[i,j] U[i*2+j, c]) * (sum_j B[j,k] V[j*2+k, c])
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    def a(i, j):
        return i * 2 + j
    def b(j, k):
        return j * 2 + k
    def c(k, i):
        return k * 2 + i

    # Strassen products
    M = [
        # M1 = (A00+A11)(B00+B11)
        ([(a(0,0),1),(a(1,1),1)], [(b(0,0),1),(b(1,1),1)]),
        # M2 = (A10+A11) B00
        ([(a(1,0),1),(a(1,1),1)], [(b(0,0),1)]),
        # M3 = A00 (B01-B11)
        ([(a(0,0),1)], [(b(0,1),1),(b(1,1),-1)]),
        # M4 = A11 (B10-B00)
        ([(a(1,1),1)], [(b(1,0),1),(b(0,0),-1)]),
        # M5 = (A00+A01) B11
        ([(a(0,0),1),(a(0,1),1)], [(b(1,1),1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(a(1,0),1),(a(0,0),-1)], [(b(0,0),1),(b(0,1),1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(a(0,1),1),(a(1,1),-1)], [(b(1,0),1),(b(1,1),1)]),
    ]
    Csum = [
        ([(0,1),(3,1),(4,-1),(6,1)], (0,0)),  # C00 = M1+M4-M5+M7
        ([(2,1),(4,1)], (0,1)),               # C01 = M3+M5
        ([(1,1),(3,1)], (1,0)),               # C10 = M2+M4
        ([(0,1),(1,-1),(2,1),(5,1)], (1,1)),  # C11 = M1-M2+M3+M6
    ]
    for terms, (k, i) in Csum:
        for mc, sign in terms:
            W[c(k,i)][mc] += sign
    for mc, (Uterms, Vterms) in enumerate(M):
        for idx, sign in Uterms:
            U[idx][mc] += sign
        for idx, sign in Vterms:
            V[idx][mc] += sign
    return U, V, W


def _verify(U, V, W, n, m, p, iters=100):
    import random
    random.seed(12345)
    rank = len(W[0])
    for _ in range(iters):
        A = [[random.randint(-4, 4) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-4, 4) for _ in range(p)] for _ in range(m)]
        C = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C[i][k] = s
        for i in range(n):
            for k in range(p):
                s = 0
                for c in range(rank):
                    ua = 0
                    for j in range(m):
                        ua += A[i][j] * U[i * m + j][c]
                    vb = 0
                    for j in range(m):
                        vb += B[j][k] * V[j * p + k][c]
                    s += W[k * n + i][c] * ua * vb
                if s != C[i][k]:
                    return False
    return True


def _build_3x4x4():
    """Rank-21 decomposition of 3x4x4 by tiling Strassen over the 3 rows of A."""
    n, m, p = 3, 4, 4
    U0, V0, W0 = _strassen_2x2x2()
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for row in range(n):
        base = row * 7
        for c in range(7):
            for idx in range(4):
                if U0[idx][c] != 0:
                    sub_i = idx // 2
                    sub_j = idx % 2
                    col = sub_j if sub_i == 0 else 2 + sub_j
                    U[row * m + col][base + c] += U0[idx][c]
            for idx in range(4):
                if V0[idx][c] != 0:
                    sub_j = idx // 2
                    sub_k = idx % 2
                    V[sub_j * p + sub_k][base + c] += V0[idx][c]
            for idx in range(4):
                if W0[idx][c] != 0:
                    sub_k = idx // 2
                    sub_i = idx % 2
                    col = sub_k
                    W[col * n + row][base + c] += W0[idx][c]
    return U, V, W


def _build_3x4x8():
    """Two independent 3x4x4 sub-tensors -> rank 42."""
    n, m, p = 3, 4, 8
    U1, V1, W1 = _build_3x4x4()
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for half in range(2):
        base = half * 21
        for i in range(n * m):
            for c in range(21):
                if U1[i][c] != 0:
                    U[i][base + c] += U1[i][c]
        for j in range(m):
            for k in range(4):
                for c in range(21):
                    if V1[j * 4 + k][c] != 0:
                        V[j * p + (k + half * 4)][base + c] += V1[j * 4 + k][c]
        for k in range(4):
            for i in range(n):
                for c in range(21):
                    if W1[k * n + i][c] != 0:
                        W[(k + half * 4) * n + i][base + c] += W1[k * n + i][c]
    return U, V, W


def _reduce_rank(U, V, W, n, m, p, deadline):
    """Greedy column-merging: if a column is a linear combination of others
    across all three factor matrices simultaneously, drop it. Returns reduced."""
    import time
    rank = len(W[0])
    keep = list(range(rank))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_keep = []
        for c in keep:
            if time.time() > deadline:
                new_keep.append(c)
                continue
            # Check if column c is in span of the other kept columns
            # Build matrix of other columns' triple-stacked vectors.
            others = [x for x in keep if x != c]
            # stacked vector length = len(U[0])*? Use U rows * V rows * W rows? 
            # Column c defines a vector in R^{n*m + m*p + p*n} (concatenation of the
            # three factor columns). It is redundant iff that vector is in the span
            # of the other columns' stacked vectors.
            L = n * m + m * p + p * n
            def stack(col):
                v = [0.0] * L
                for r in range(n * m):
                    v[r] = U[r][col]
                for r in range(m * p):
                    v[n * m + r] = V[r][col]
                for r in range(p * n):
                    v[n * m + m * p + r] = W[r][col]
                return v
            # Gaussian elimination to test membership
            mat = [stack(x) for x in others]
            # row-reduce augmented with target
            target = stack(c)
            # Build augmented matrix rows = mat, test if target in column span
            # Use numpy-free elimination on the transpose: columns are vectors.
            # Simpler: solve mat^T x = target (least squares exact check).
            # Use simple Gaussian elimination on system M^T x = t.
            A_ = [[mat[r][col_idx] for r in range(len(mat))] for col_idx in range(L)]
            b_ = [target[col_idx] for col_idx in range(L)]
            # Row-reduce [A_ | b_]
            nrows = len(A_)
            ncols = len(mat)
            M = [row[:] + [b_] for row in A_]
            pivot_row = 0
            for col in range(ncols):
                if time.time() > deadline:
                    return None
                piv = None
                for r in range(pivot_row, nrows):
                    if abs(M[r][col]) > 1e-9:
                        piv = r
                        break
                if piv is None:
                    continue
                M[pivot_row], M[piv] = M[piv], M[pivot_row]
                pv = M[pivot_row][col]
                for r in range(nrows):
                    if r != pivot_row and abs(M[r][col]) > 1e-12:
                        f = M[r][col] / pv
                        for cc in range(col, ncols + 1):
                            M[r][cc] -= f * M[pivot_row][cc]
                pivot_row += 1
                if pivot_row == nrows:
                    break
            # Check consistency
            consistent = True
            for r in range(nrows):
                if all(abs(M[r][cc]) < 1e-9 for cc in range(ncols)) and abs(M[r][ncols]) > 1e-9:
                    consistent = False
                    break
            if consistent:
                # column c is redundant -> drop
                changed = True
            else:
                new_keep.append(c)
        keep = new_keep
    if len(keep) >= rank:
        return (U, V, W)
    r = len(keep)
    U2 = [[U[rr][keep[c]] for c in range(r)] for rr in range(n * m)]
    V2 = [[V[rr][keep[c]] for c in range(r)] for rr in range(m * p)]
    W2 = [[W[rr][keep[c]] for c in range(r)] for rr in range(p * n)]
    return (U2, V2, W2)


def construct():
    import time
    start = time.time()
    deadline = start + 115.0

    n, m, p = DIMENSIONS
    U, V, W = _build_3x4x8()

    if not _verify(U, V, W, n, m, p, iters=80):
        # Fallback: guaranteed schoolbook
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

    if time.time() < deadline:
        reduced = _reduce_rank(U, V, W, n, m, p, deadline)
        if reduced is not None and reduced[0] is not U:
            U, V, W = reduced
            if _verify(U, V, W, n, m, p, iters=30):
                pass
            else:
                U, V, W = _build_3x4x8()

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
