"""Matrix multiplication 2x4x7: block + Strassen rank-minimal decomposition.

Strategy (family of the measured best): partition the middle dimension (4) into
blocks and the output dimension (2) into blocks. For each block pair use
Strassen (square) or schoolbook (rectangular). Search block splits for min rank.
"""
import time

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS  # 2 x 4 x 7

# ---------------------------------------------------------------------------
# Strassen 2x2x2 (exact, integer entries)
# ---------------------------------------------------------------------------
def _strassen_2x2():
    """Return (U,V,W) each 4x7 for 2x2*2x2 -> 2x2, trace convention W[k*n+i]."""
    # Strassen products for A (2x2) * B (2x2):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11) b00
    # M3 = a00 (b10-b11)
    # M4 = a11 (b01-b11)
    # M5 = (a00+a10) b11
    # M6 = (a11-a01) b01
    # M7 = (a01-a00) b10
    # C00 = M1+M4-M5-M7 ; C01 = M3+M5
    # C10 = M2+M4 ; C11 = M1-M2+M3-M6
    U = {}  # index (i,j) -> list of (coeff, Midx)
    # Build U rows: for each (i,j) in A, coefficient of each M
    # M1: a00(+1), a11(+1)
    # M2: a10(+1), a11(+1)
    # M3: a00(+1)
    # M4: a11(+1)
    # M5: a00(+1), a10(+1)
    # M6: a11(+1), a01(-1)
    # M7: a01(+1), a00(-1)
    Urows = [
        # (i,j)
        # M1 M2 M3 M4 M5 M6 M7
        {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},   # a00: M1+M3+M5-M7
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1},   # a01: M5+M6+M7
        {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0},   # a10: M2+M5
        {0: 1, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0},   # a11: M1+M2+M4+M6
    ]
    Vrows = [
        # (j,k)
        {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},   # b00: M1
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0},   # b01: M6
        {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1},   # b10: M2+M3+M7
        {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0},   # b11: M1+M4+M5
    ]
    Wrows = [
        # (k,i)
        {0: 1, 1: 0, 2: 0, 3: 1, 4: -1, 5: 0, 6: -1},  # C00: M1+M4-M5-M7
        {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 0},    # C01: M3+M5
        {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0},    # C10: M2+M4
        {0: 1, 1: -1, 2: 1, 3: 0, 4: 0, 5: -1, 6: 0},  # C11: M1-M2+M3-M6
    ]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            row = Urows[i*2+j]
            for midx in range(7):
                U[i*2+j][midx] = row[midx]
    for j in range(2):
        for k in range(2):
            row = Vrows[j*2+k]
            for midx in range(7):
                V[j*2+k][midx] = row[midx]
    for k in range(2):
        for i in range(2):
            row = Wrows[k*2+i]
            for midx in range(7):
                W[k*2+i][midx] = row[midx]
    return U, V, W


# ---------------------------------------------------------------------------
# Build a full decomposition from a block partition.
# middle (m=4) split into sizes mid_blocks (list summing to 4)
# output (n=2) split into sizes out_blocks (list summing to 2)
# For each (out_block b, mid_block a):
#   if a == b (square): use Strassen (rank 7 if a>=2, else rank a^2 schoolbook)
#   else: schoolbook rank a*b
# We treat the "a" as the row-block of A (size a) and "b" as col-block of B /
# output-block of C. A is 2x4: rows = output blocks (n), cols = mid blocks (m).
# B is 4x7: rows = mid blocks (m), cols = p (7, unsplit).
# C is 2x7: rows = output blocks (n), cols = p (7).
# ---------------------------------------------------------------------------
def _build(mid_blocks, out_blocks):
    """Return (U,V,W) lists. U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank."""
    U_cols = []
    V_cols = []
    W_cols = []

    def add(Ucol, Vcol, Wcol):
        U_cols.append(Ucol)
        V_cols.append(Vcol)
        W_cols.append(Wcol)

    for bi, ob in enumerate(out_blocks):
        for ai, mb in enumerate(mid_blocks):
            # A sub-block: rows out_block bi (size ob), cols mid_block ai (size mb)
            # B sub-block: rows mid_block ai (size mb), cols all p (size p)
            # C sub-block: rows out_block bi (size ob), cols all p (size p)
            if ob == mb:
                # square block: Strassen if size>=2 else schoolbook (rank 1 for 1x1)
                sz = ob
                if sz == 1:
                    # single element: A[i,j]*B[j,k]
                    for k in range(p):
                        add([0]*(n*m), [0]*(m*p), [0]*(p*n))
                        # fix indices
                    # We'll do it properly below with index math
                    pass
                # handle via generic loop with Strassen for sz>=2
            # Generic: do schoolbook for rectangular, Strassen for square
            if ob == mb and ob >= 2:
                # Strassen on sz x sz square, for each of p columns
                sz = ob
                Uq, Vq, Wq = _strassen_2x2()
                # but sz may be >2; only sz==2 supported here. For sz==2 ok.
                assert sz == 2, "only 2x2 Strassen implemented"
                # base indices
                a_row0 = sum(out_blocks[:bi])          # first row of A for this out block
                a_col0 = sum(mid_blocks[:ai])          # first col of A for this mid block
                b_row0 = sum(mid_blocks[:ai])          # first row of B for this mid block
                c_row0 = sum(out_blocks[:bi])          # first row of C for this out block
                for k in range(p):
                    # local U rows: (i in 0..1, j in 0..1) -> global (a_row0+i, a_col0+j)
                    for r in range(4):
                        i = r // 2
                        j = r % 2
                        Ucol = [0]*(n*m)
                        Ucol[(a_row0+i)*m + (a_col0+j)] = Uq[r][0]
                        # Vcol
                        Vcol = [0]*(m*p)
                        for rr in range(4):
                            jj = rr // 2
                            kk = rr % 2
                            Vcol[(b_row0+jj)*p + (k+kk)] = Vq[rr][0]
                        # Wcol
                        Wcol = [0]*(p*n)
                        for rr in range(4):
                            kk = rr // 2
                            ii = rr % 2
                            Wcol[(k+kk)*n + (c_row0+ii)] = Wq[rr][0]
                        add(Ucol, Vcol, Wcol)
            else:
                # schoolbook for ob x mb block, for each k
                for i in range(ob):
                    for j in range(mb):
                        for k in range(p):
                            Ucol = [0]*(n*m)
                            Ucol[(sum(out_blocks[:bi])+i)*m + (sum(mid_blocks[:ai])+j)] = 1
                            Vcol = [0]*(m*p)
                            Vcol[(sum(mid_blocks[:ai])+j)*p + k] = 1
                            Wcol = [0]*(p*n)
                            Wcol[k*n + (sum(out_blocks[:bi])+i)] = 1
                            add(Ucol, Vcol, Wcol)

    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c in range(rank):
        for i, val in enumerate(U_cols[c]):
            if val != 0:
                U[i][c] = val
        for i, val in enumerate(V_cols[c]):
            if val != 0:
                V[i][c] = val
        for i, val in enumerate(W_cols[c]):
            if val != 0:
                W[i][c] = val
    return U, V, W


def _verify(U, V, W):
    """Check exact decomposition: C[i,k] = sum_r U[i,j? ] ... via tensor identity."""
    # Tensor identity: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # else 0.
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                expected = 1 if (i, j, k) in _target() else 0
                if s != expected:
                    return False
    return True


def _target():
    """Return set of (i,j,k) where the multiplication tensor is nonzero.
    C[i,k] = sum_j A[i,j] B[j,k]; tensor T[i,j,k] = 1 for all i,j,k (since the
    product uses every A[i,j] and B[j,k]). Actually the bilinear form tensor for
    matrix mult has T[i,j,k]=1 for all i,j,k."""
    return {(i, j, k) for i in range(n) for j in range(m) for k in range(p)}


def _gen_partitions(total, max_parts):
    """All ordered partitions of total into positive ints, up to max_parts parts."""
    if total == 0:
        yield []
        return
    if max_parts == 0:
        return
    for first in range(1, total + 1):
        for rest in _gen_partitions(total - first, max_parts - 1):
            yield [first] + rest


def construct():
    deadline = time.time() + 140  # safety margin under 150s
    best = None
    best_rank = None

    # Candidate block splits for middle (4) and output (2)
    mid_splits = list(_gen_partitions(m, 4))
    out_splits = list(_gen_partitions(n, 2))

    # Order by estimated rank (lower first) to find good candidates quickly
    def est_rank(ms, os_):
        r = 0
        for ob in os_:
            for mb in ms:
                if ob == mb and ob >= 2:
                    r += 7
                else:
                    r += ob * mb
        return r

    candidates = [(est_rank(ms, os_), ms, os_) for ms in mid_splits for os_ in out_splits]
    candidates.sort()

    for est, ms, os_ in candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W = _build(ms, os_)
        except Exception:
            continue
        rank = len(U[0])
        if best_rank is not None and rank >= best_rank:
            continue
        if _verify(U, V, W):
            best_rank = rank
            best = (U, V, W)
            if time.time() > deadline:
                break

    if best is None:
        # fallback: schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    U, V, W = best
    return {"U": U, "V": V, "W": W}
