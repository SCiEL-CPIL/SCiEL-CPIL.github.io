"""Matrix multiplication 2x3x7 rank-minimal search via folding from schoolbook."""
import time

DIMENSIONS = [2, 3, 7]
DEADLINE = time.time() + 130  # safety margin under 150s

def _schoolbook(n, m, p):
    """Return (U, V, W) as dicts {(i,j):...} no, use list-of-lists with term indexing."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # term t corresponds to (i,j,k)
    idx = 0
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                terms.append((i, j, k))
                idx += 1
    return U, V, W, terms

def _verify(U, V, W, n, m, p):
    """Check tensor decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    and for all (i,j,k) with not (i,j,k) a valid triple... actually the tensor T[i,j,k] = 1 for all i,j,k.
    The condition is: for every (i,j,k) in [n]x[m]x[p], sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _rank_of(U, V, W):
    """Effective rank: number of columns where not all three are zero (and that contribute)."""
    # We count columns that are non-zero in at least one of U,V,W rows that matter.
    # Simpler: count columns where any entry is non-zero.
    rank = 0
    nc = len(U[0])
    for c in range(nc):
        nz = False
        for r in range(len(U)):
            if U[r][c] != 0:
                nz = True
                break
        if not nz:
            for r in range(len(V)):
                if V[r][c] != 0:
                    nz = True
                    break
        if not nz:
            for r in range(W):
                if W[r][c] != 0:
                    nz = True
                    break
        if nz:
            rank += 1
    return rank

def _apply_fold(U, V, W, t1, t2, terms):
    """Fold terms t1 and t2. Returns new U,V,W,terms or None if invalid.
    t1=(i1,j1,k1), t2=(i2,j2,k2) with i1!=i2 and j1!=j2.
    Replace with (i1,j1,k2) and (i2,j2,k1).
    """
    i1, j1, k1 = terms[t1]
    i2, j2, k2 = terms[t2]
    if i1 == i2 or j1 == j2:
        return None
    # New terms
    new_terms = list(terms)
    new_terms[t1] = (i1, j1, k2)
    new_terms[t2] = (i2, j2, k1)
    # Rebuild matrices from scratch is expensive; instead do in-place:
    # Zero out old contributions, add new.
    n = len(U) // 3  # actually n*m = len(U)
    # We need n,m,p. Pass them.
    return None  # placeholder

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    U, V, W, terms = _schoolbook(n, m, p)

    # Greedy folding: repeatedly find a valid pair (t1,t2) with i1!=i2, j1!=j2,
    # apply the fold (which keeps the same number of terms but changes structure),
    # then try to drop any column that becomes zero.
    # Actually folding does not reduce term count. The rank reduction comes from
    # the fact that after folding, some columns may combine or become redundant.
    #
    # Better approach: the folding operation on the *set of monomials* doesn't change
    # count. To reduce rank, we need a different technique.
    #
    # Let me reconsider. The standard way to reduce bilinear complexity is via
    # "bilinear folding" which actually CAN reduce rank when applied to the
    # tensor as a whole, because it changes which entries are non-zero in a way
    # that allows cancellation.
    #
    # For 2x3x7, known results: the bilinear complexity is at most 42 (schoolbook)
    # and lower bounds exist. Let me just do a search over fold sequences and
    # check if any column becomes zero.

    # Simpler: try all single folds, check if any column becomes zero, repeat.
    def try_fold(U, V, W, terms, t1, t2, n, m, p):
        i1, j1, k1 = terms[t1]
        i2, j2, k2 = terms[t2]
        if i1 == i2 or j1 == j2:
            return None
        # Make copies
        U2 = [row[:] for row in U]
        V2 = [row[:] for row in V]
        W2 = [row[:] for row in W]
        terms2 = list(terms)
        # Zero out t1, t2
        for c in (t1, t2):
            U2[i1*m+j1][c] = 0
            V2[j1*p+k1][c] = 0
            W2[k1*n+i1][c] = 0
            U2[i2*m+j2][c] = 0
            V2[j2*p+k2][c] = 0
            W2[k2*n+i2][c] = 0
        # Add new terms
        U2[i1*m+j1][t1] = 1
        V2[j1*p+k2][t1] = 1
        W2[k2*n+i1][t1] = 1
        U2[i2*m+j2][t2] = 1
        V2[j2*p+k1][t2] = 1
        W2[k1*n+i2][t2] = 1
        terms2[t1] = (i1, j1, k2)
        terms2[t2] = (i2, j2, k1)
        return U2, V2, W2, terms2

    # Greedy: keep applying folds that reduce effective rank
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        cur_rank = _rank_of(U, V, W)
        for t1 in range(len(terms)):
            if time.time() > DEADLINE:
                break
            for t2 in range(t1 + 1, len(terms)):
                res = try_fold(U, V, W, terms, t1, t2, n, m, p)
                if res is None:
                    continue
                U2, V2, W2, terms2 = res
                new_rank = _rank_of(U2, V2, W2)
                if new_rank < cur_rank:
                    U, V, W, terms = U2, V2, W2, terms2
                    cur_rank = new_rank
                    improved = True
                    break
            if improved:
                break

    # Final: compact to remove zero columns
    rank = _rank_of(U, V, W)
    # Build compact version
    keep = []
    for c in range(len(U[0])):
        nz = any(U[r][c] != 0 for r in range(len(U))) or \
             any(V[r][c] != 0 for r in range(len(V))) or \
             any(W[r][c] != 0 for r in range(len(W)))
        if nz:
            keep.append(c)
    Uf = [[U[r][c] for c in keep] for r in range(len(U))]
    Vf = [[V[r][c] for c in keep] for r in range(len(V))]
    Wf = [[W[r][c] for c in keep] for r in range(len(W))]

    # Verify
    if not _verify(Uf, Vf, Wf, n, m, p):
        # Fallback to schoolbook
        U, V, W, terms = _schoolbook(n, m, p)
        Uf = [row[:] for row in U]
        Vf = [row[:] for row in V]
        Wf = [row[:] for row in W]

    return {"U": Uf, "V": Vf, "W": Wf}
