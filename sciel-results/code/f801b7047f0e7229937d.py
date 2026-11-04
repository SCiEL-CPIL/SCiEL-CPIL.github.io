"""Fresh design: rank-minimal exact tensor decomposition for 4x4 * 4x5."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

_DEADLINE = time.time() + 130.0  # 130s wall-clock guard (10s safety margin vs 140s kill)

def _time_left():
    return _DEADLINE - time.time()

def _tensor(n, m, p):
    """Multiplication tensor T[i,j,k] = 1 for the (i,j) input, (k) output pairing.
    We build the 3-way array T[a,b,c] where a indexes A (n*m), b indexes B (m*p),
    c indexes C (p*n). T[a,b,c] = 1 iff the product entry (i,k) with i=A-row,
    k=C-col uses A[i,j]*B[j,k]."""
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _reconstruct(U, V, W, n, m, p):
    """Reconstruct tensor from factors (lists). U shape (n*m, r), V (m*p, r), W (p*n, r)."""
    r = len(U[0]) if U else 0
    T = np.zeros((n*m, m*p, p*n), dtype=object)
    for c in range(r):
        # outer product of U[:,c], V[:,c], W[:,c]
        u = [U[a][c] for a in range(n*m)]
        v = [V[b][c] for b in range(m*p)]
        w = [W[cc][c] for cc in range(p*n)]
        for a in range(n*m):
            if u[a] == 0:
                continue
            for b in range(m*p):
                if v[b] == 0:
                    continue
                val = u[a]*v[b]
                if val == 0:
                    continue
                for cc in range(p*n):
                    if w[cc] != 0:
                        T[a,b,cc] = T[a,b,cc] + val*w[cc]
    return T

def _is_exact(T, n, m, p):
    """Check T equals the multiplication tensor exactly (integer equality)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if T[a,b,c] != 1:
                    return False
    # all other entries must be 0
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # skip the 1-entries already checked
                # find if (a,b,c) is a product entry
                i, j = divmod(a, m)
                jj, kk = divmod(b, p)
                kk2, ii = divmod(c, n)
                if i == ii and j == jj and j == kk and k == kk2:
                    continue
                if T[a,b,c] != 0:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _random_factors(n, m, p, r, seed):
    """Random integer factors in {-1,0,1} (mostly 0 for sparsity)."""
    rng = random.Random(seed)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    for c in range(r):
        # pick a sparse support per factor column
        for a in range(n*m):
            if rng.random() < 0.35:
                U[a][c] = rng.choice([-1,1])
        for b in range(m*p):
            if rng.random() < 0.35:
                V[b][c] = rng.choice([-1,1])
        for cc in range(p*n):
            if rng.random() < 0.35:
                W[cc][c] = rng.choice([-1,1])
    return U, V, W

def _try_shrink(U, V, W, n, m, p, max_iters, rng):
    """Local search: try flipping/removing entries to reduce rank while staying exact.
    Returns best (U,V,W) with smaller rank if found, else None."""
    cur_r = len(U[0])
    best = None
    for it in range(max_iters):
        if _time_left() < 5:
            break
        # pick a random column and try zeroing it (rank reduction)
        c = rng.randrange(cur_r)
        # check if column c is redundant: removing it keeps exactness?
        U2 = [row[:c] + row[c+1:] for row in U]
        V2 = [row[:c] + row[c+1:] for row in V]
        W2 = [row[:c] + row[c+1:] for row in W]
        if _is_exact(_reconstruct(U2, V2, W2, n, m, p), n, m, p):
            U, V, W, cur_r = U2, V2, W2, cur_r - 1
            best = (U, V, W)
            continue
        # try flipping a single entry in column c of one factor
        for which in range(3):
            if _time_left() < 5:
                break
            F = [U, V, W][which]
            sz = [len(U), len(V), len(W)][which]
            a = rng.randrange(sz)
            old = F[a][c]
            if old == 0:
                F[a][c] = rng.choice([-1,1])
            else:
                F[a][c] = -old
            T2 = _reconstruct(U, V, W, n, m, p)
            if _is_exact(T2, n, m, p):
                # keep the flip
                best = ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W])
            else:
                F[a][c] = old
    return best

def construct():
    n, m, p = DIMENSIONS
    rng = random.Random(12345)

    # Start from schoolbook (always valid)
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    best = (U, V, W)

    # Try a range of ranks below schoolbook, from low to high.
    # For each rank, random restarts + local shrink.
    target_ranks = list(range(17, best_rank))  # 17..79
    for r in target_ranks:
        if _time_left() < 10:
            break
        # a few restarts per rank
        restarts = 3 if _time_left() > 30 else 1
        for s in range(restarts):
            if _time_left() < 10:
                break
            U, V, W = _random_factors(n, m, p, r, rng.randrange(10**9))
            # quick exactness check; if exact, great
            if _is_exact(_reconstruct(U, V, W, n, m, p), n, m, p):
                best = (U, V, W)
                best_rank = r
                break
            # local search to try to fix/shrink
            shrunk = _try_shrink(U, V, W, n, m, p, max_iters=200, rng=rng)
            if shrunk is not None:
                U2, V2, W2 = shrunk
                rr = len(U2[0])
                if rr < best_rank:
                    best = (U2, V2, W2)
                    best_rank = rr
    # Also attempt to shrink the schoolbook decomposition itself
    shrunk = _try_shrink(best[0], best[1], best[2], n, m, p, max_iters=400, rng=rng)
    if shrunk is not None:
        U2, V2, W2 = shrunk
        if len(U2[0]) < best_rank:
            best = (U2, V2, W2)
            best_rank = len(U2[0])

    U, V, W = best
    return {"U": U, "V": V, "W": W}
