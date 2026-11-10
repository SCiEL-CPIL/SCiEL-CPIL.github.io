# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Small Nikodym set in F_49^2, F_49 = F_7[alpha], alpha^2 = 3.

    A set N is Nikodym iff for every point x there is a direction v in
    {(1, m): m in F_49} U {(0,1)} such that the punctured line
    {x + t v : t != 0} is fully contained in N.  We maximize the complement
    (points not in N) with fast numpy local search over diverse algebraic
    seeds, guarded by a wall-clock deadline.
    """
    t0 = time.time()
    DEADLINE = t0 + 560.0

    p = 7
    k = 3
    q = 49
    npts = q * q
    nlines = q + 1

    def add(x, y):
        return (x // p + y // p) % p + ((x % p + y % p) % p) * p

    def mul(x, y):
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        return (xa * ya + xb * yb * k) % p + ((xa * yb + xb * ya) % p) * p

    def pt_idx(xa, xb, ya, yb):
        return xa + xb * p + (ya + yb * p) * q

    def unpack(p):
        xa = p % p
        t = p // p
        xb = t % p
        t //= p
        ya = t % p
        yb = t // p
        return xa, xb, ya, yb

    # ---- line tables ------------------------------------------------------
    line = np.empty((nlines, npts, q - 1), dtype=np.int32)
    line_through = np.empty((npts, nlines, q), dtype=np.int32)
    for d in range(nlines):
        vert = (d == q)
        m = (None if vert else d)
        for p in range(npts):
            xa, xb, ya, yb = unpack(p)
            arr = np.empty(q, dtype=np.int32)
            for t in range(q):
                if vert:
                    cx_a, cx_b = xa, xb
                    cy_a, cy_b = ya, (yb + t) % p
                else:
                    # x = x + t   (field addition)
                    nx = add(p // q * 0 + (xa + xb * p), t)  # x as field elem
                    nx = add(xa + xb * p, t)
                    cx_a, cx_b = nx // p, nx % p
                    # y = y + t*m
                    tm = mul(t, m)
                    ny = add(ya + yb * p, tm)
                    cy_a, cy_b = ny // p, ny % p
                arr[t] = pt_idx(cx_a, cx_b, cy_a, cy_b)
            line_through[p, d, :] = arr
            line[d, p, :] = arr[arr != p]

    def is_valid(N):
        covered = np.zeros(npts, dtype=bool)
        for d in range(nlines):
            covered |= N[line[d, :, :]].all(axis=1)
        return bool(covered.all())

    def initial_missing(N):
        notN = ~N
        missing = np.empty((nlines, npts), dtype=np.int32)
        for d in range(nlines):
            missing[d, :] = notN[line[d, :, :]].sum(axis=1)
        return missing

    def local_search(N, missing, max_iters=500000):
        inN = list(np.where(N)[0])
        for _ in range(max_iters):
            if time.time() > DEADLINE - 30:
                break
            if not inN:
                break
            random.shuffle(inN)
            made = False
            for p in inN:
                N[p] = False
                for d in range(nlines):
                    missing[d, line_through[p, d, :]] += 1
                if np.any(np.all(missing > 0, axis=0)):
                    N[p] = True
                    for d in range(nlines):
                        missing[d, line_through[p, d, :]] -= 1
                    continue
                inN.remove(p)
                made = True
                break
            if not made:
                break
        return N

    def seed_from_yset(yset):
        N = np.ones(npts, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        if (2 * xa + ya * ya - yb * yb * k) % p == 0 and (ya, yb) not in yset:
                            N[pt_idx(xa, xb, ya, yb)] = False
        return N

    yA = {(u * u % p, v) for u in range(p) for v in range(p)}
    yB = {(u, v * v % p) for u in range(p) for v in range(p)}
    yC = {(u * u % p, v * v % p) for u in range(p) for v in range(p)}
    seeds = [yA, yB, yC]

    best_N = None
    best_comp = -1
    for yset in seeds:
        if time.time() > DEADLINE - 60:
            break
        N = seed_from_yset(yset)
        if not is_valid(N):
            continue
        missing = initial_missing(N)
        N = local_search(N, missing)
        comp = npts - int(N.sum())
        if comp > best_comp:
            best_comp = comp
            best_N = N.copy()

    if best_N is None:
        best_N = np.ones(npts, dtype=bool)

    pts_in_N = np.where(best_N)[0]
    out = np.empty((len(pts_in_N), 2, 2), dtype=np.int64)
    for i, p in enumerate(pts_in_N):
        xa, xb, ya, yb = unpack(p)
        out[i, 0, 0] = xa
        out[i, 0, 1] = xb
        out[i, 1, 0] = ya
        out[i, 1, 1] = yb
    return out

# EVOLVE-BLOCK-END
