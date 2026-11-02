"""Rewrite: constructive pool + vectorized greedy + local swaps for kissing number in R^11."""
import time
import numpy as np

D = 11
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _is_valid(V):
    """Return True if every pairwise distance >= max norm, all nonzero."""
    if len(V) == 0:
        return True
    norms = np.linalg.norm(V, axis=1)
    if np.any(norms <= 0):
        return False
    maxn = norms.max()
    Dm = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    np.fill_diagonal(Dm, np.inf)
    return bool(Dm.min() >= maxn - 1e-9)


def _build_pool():
    """Generate a large, diverse pool of nonzero integer vectors."""
    pool = []
    # 1) Vectors with exactly 2 nonzero coords: (±1,±1) on any pair
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    pool.append(v)
    # 2) Vectors with exactly 3 nonzero coords: (±1,±1,±1) on any triple
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for s in (1, -1):
                    for t in (1, -1):
                        for u in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = s
                            v[j] = t
                            v[k] = u
                            pool.append(v)
    # 3) Some vectors with a 2 and a 1: (±2, ±1) on pairs, and (±2,±1,±1) on triples
    for i in range(D):
        for j in range(i + 1, D):
            for si in (2, -2):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    pool.append(v)
                    pool.append(v * -1)
    # 4) Coordinate unit vectors times 2 (norm 2) - helps padding
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2
            v *= s
            pool.append(v)
    # 5) Random small integer vectors for diversity
    rng = np.random.default_rng(12345)
    for _ in range(20000):
        nz = rng.integers(2, 5)  # 2..4 nonzero
        idx = rng.choice(D, size=nz, replace=False)
        vals = rng.choice([1, 2, -1, -2], size=nz)
        v = np.zeros(D, dtype=np.int64)
        v[idx] = vals
        if np.all(v == 0):
            continue
        pool.append(v)
    arr = np.array(pool, dtype=np.int64)
    # Remove exact duplicates and zero vectors
    arr = arr[np.linalg.norm(arr, axis=1) > 0]
    # Unique rows
    _, idx = np.unique(arr, axis=0, return_index=True)
    arr = arr[np.sort(idx)]
    return arr


def _greedy_select(pool, max_iter=2000000):
    """Vectorized-ish greedy: repeatedly pick a vector compatible with all chosen."""
    chosen = []
    norms = np.linalg.norm(pool, axis=1)
    maxn = 0.0
    # Precompute pool-pool distances once (memory heavy if pool huge; cap it)
    n = len(pool)
    if n > 200000:
        # subsample to keep distance matrix manageable
        sel = np.random.default_rng(0).choice(n, size=200000, replace=False)
        pool = pool[sel]
        norms = norms[sel]
        n = 200000
    # Distance matrix
    Dm = np.linalg.norm(pool[:, None, :] - pool[None, :, :], axis=2)
    np.fill_diagonal(Dm, np.inf)
    # Greedy with random restarts handled outside; here deterministic order by norm
    order = np.argsort(norms)
    used = np.zeros(n, dtype=bool)
    for idx in order:
        if _elapsed() > _DEADLINE - 5:
            break
        v = pool[idx]
        nn = norms[idx]
        # check against all chosen
        ok = True
        for c in chosen:
            d = np.linalg.norm(v - c)
            if d < maxn - 1e-9 or d < nn - 1e-9:
                ok = False
                break
        if ok:
            chosen.append(v)
            if nn > maxn:
                maxn = nn
    return np.array(chosen, dtype=np.int64) if chosen else np.zeros((0, D), dtype=np.int64)


def _local_swap(V, pool, iters=2000):
    """Try to add vectors from pool that are compatible, and swap out incompatible chosen ones."""
    if len(V) == 0:
        return V
    norms = np.linalg.norm(V, axis=1)
    maxn = norms.max()
    Vlist = list(V)
    improved = True
    it = 0
    while improved and it < iters and _elapsed() < _DEADLINE - 5:
        improved = False
        it += 1
        # Try to add each pool vector not already in V
        for p in pool:
            if _elapsed() > _DEADLINE - 5:
                break
            pn = np.linalg.norm(p)
            if pn > maxn + 1e-9:
                continue
            # check compatibility with all current
            ok = True
            for c in Vlist:
                d = np.linalg.norm(p - c)
                if d < maxn - 1e-9:
                    ok = False
                    break
            if ok:
                Vlist.append(p)
                improved = True
    return np.array(Vlist, dtype=np.int64)


def construct():
    start = time.time()
    deadline = start + _DEADLINE - 5  # safety margin
    pool = _build_pool()
    best = None
    best_k = 0
    # A few greedy passes with different orderings
    for trial in range(6):
        if time.time() > deadline:
            break
        if trial == 0:
            V = _greedy_select(pool)
        else:
            # shuffle pool order
            idx = np.random.default_rng(trial * 7 + 1).permutation(len(pool))
            V = _greedy_select(pool[idx])
        if len(V) > best_k:
            best = V
            best_k = len(V)
        # local swap to add more
        if time.time() < deadline:
            V2 = _local_swap(V, pool, iters=500)
            if len(V2) > best_k:
                best = V2
                best_k = len(V2)
    if best is None or len(best) == 0:
        # fallback: seed-like 2-nonzero vectors
        import itertools
        vecs = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vecs.append(v)
        best = np.array(vecs, dtype=np.int64)
    # Final validity check; if invalid, fall back to a safe subset
    if not _is_valid(best):
        # trim greedily to a valid subset
        norms = np.linalg.norm(best, axis=1)
        maxn = norms.max()
        kept = []
        for v in best:
            ok = True
            for c in kept:
                if np.linalg.norm(v - c) < maxn - 1e-9:
                    ok = False
                    break
            if ok:
                kept.append(v)
        best = np.array(kept, dtype=np.int64)
    return best
