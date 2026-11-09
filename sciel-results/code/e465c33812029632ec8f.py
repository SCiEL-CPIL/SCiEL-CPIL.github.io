"""Fresh design: fast local-search colouring of K_n with 4 colours, no monochromatic triangle.
Entry point construct() -> {'n': n, 'colors': [...]} (lexicographic pairs i<j, colours in {0,1,2,3}).
"""
import time
import random
import numpy as np

_DEADLINE = 145.0  # safety margin under the 150 s wall-clock kill
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _base_32():
    # Greenwood-Gleason colouring of K16 (Clebsch graph x3), doubled with colour 3 between copies.
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    n = 32
    C = np.full((n, n), -1, dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            c = log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
            C[i, j] = C[j, i] = c
    return C


def _add_vertex(C):
    """Try to add one vertex; return new C or None if impossible."""
    n = C.shape[0]
    for c in range(4):
        # candidate neighbours in colour c must be independent
        mask = (C[:n, :n] == c)
        rows = np.where(mask.any(axis=1))[0]
        bad = False
        for r in rows:
            # any edge among rows[r]'s colour-c neighbours?
            nb = np.where(mask[r])[0]
            if len(nb) and np.any(mask[r, nb]):
                bad = True
                break
        if bad:
            continue
        # feasible: build new matrix
        C2 = np.full((n + 1, n + 1), -1, dtype=np.int8)
        C2[:n, :n] = C
        C2[n, :n] = c
        C2[:n, n] = c
        return C2
    return None


def _try_adds(C, max_adds):
    for _ in range(max_adds):
        C2 = _add_vertex(C)
        if C2 is None:
            break
        C = C2
    return C


def _random_moves(C, budget):
    """Random recolour/swap moves to escape local optima."""
    n = C.shape[0]
    t0 = time.time()
    while time.time() - t0 < budget:
        i, j = random.sample(range(n), 2)
        old = C[i, j]
        new = random.randrange(4)
        if new == old:
            continue
        # check no triangle in colour 'new'
        ok = True
        for k in range(n):
            if k != i and k != j and C[i, k] == new and C[j, k] == new:
                ok = False
                break
        if ok:
            C[i, j] = C[j, i] = new
    return C


def _search(start, seed):
    rng = random.Random(seed)
    best = start
    while _time_left() > 3.0:
        C = start.copy()
        # random perturbation
        n = C.shape[0]
        for _ in range(rng.randrange(0, 3 * n)):
            i, j = rng.sample(range(n), 2)
            old = C[i, j]
            new = rng.randrange(4)
            if new == old:
                continue
            ok = True
            for k in range(n):
                if k != i and k != j and C[i, k] == new and C[j, k] == new:
                    ok = False
                    break
            if ok:
                C[i, j] = C[j, i] = new
        # greedy add
        C = _try_adds(C, 30)
        # local improvement: try random moves + add
        improved = True
        while improved and _time_left() > 3.0:
            improved = False
            C2 = _random_moves(C, 0.15)
            C3 = _try_adds(C2, 30)
            if C3.shape[0] > C.shape[0]:
                C = C3
                improved = True
        if C.shape[0] > best.shape[0]:
            best = C
    return best


def construct():
    base = _base_32()
    base = _try_adds(base, 40)  # deterministic first push
    best = base
    seed = 12345
    while _time_left() > 5.0:
        cand = _search(base, seed)
        if cand.shape[0] > best.shape[0]:
            best = cand
        seed += 1
    n = best.shape[0]
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(best[i, j]))
    return {'n': n, 'colors': colors}
