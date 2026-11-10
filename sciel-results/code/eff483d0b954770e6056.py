import numpy as np
import time
import random

_N = 30
_MINC, _MAXC = -30, 30

def _init_indices():
    i, j, k = np.meshgrid(range(_N), range(_N), range(_N), indexing='ij')
    m = (i < j) & (j < k)
    return i[m], j[m], k[m]

_I, _J, _K = _init_indices()

def _doubled(P):
    x1 = P[_I, 0]; y1 = P[_I, 1]
    x2 = P[_J, 0]; y2 = P[_J, 1]
    x3 = P[_K, 0]; y3 = P[_K, 1]
    return np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def _score(P):
    d = _doubled(P)
    d = d[d > 0]
    if d.size == 0:
        return 0
    _, c = np.unique(d, return_counts=True)
    return int(c.max())

def _seed_grid():
    pts = [(a * 10, b * 10) for a in range(6) for b in range(5)]
    return np.array(pts, dtype=np.int64)

def _seed_random(rng):
    pts = set()
    while len(pts) < _N:
        pts.add((rng.randint(_MINC, _MAXC), rng.randint(_MINC, _MAXC)))
    return np.array(list(pts), dtype=np.int64)

def _seed_twoline():
    pts = [(a * 4 - 28, 0) for a in range(15)] + [(a * 4 - 28, 2) for a in range(15)]
    return np.array(pts, dtype=np.int64)

def _anneal(P0, budget, rng):
    P = P0.astype(np.int64).copy()
    pset = set(map(tuple, P.tolist()))
    cur_s = _score(P)
    best = P.copy()
    best_s = cur_s
    T = 3.0
    t0 = time.time()
    while time.time() - t0 < budget:
        i = rng.randrange(_N)
        ox, oy = int(P[i, 0]), int(P[i, 1])
        if rng.random() < 0.5:
            nx = rng.randint(_MINC, _MAXC)
            ny = rng.randint(_MINC, _MAXC)
        else:
            nx = max(_MINC, min(_MAXC, ox + rng.randint(-3, 3)))
            ny = max(_MINC, min(_MAXC, oy + rng.randint(-3, 3)))
        if (nx, ny) == (ox, oy) or (nx, ny) in pset:
            continue
        pset.discard((ox, oy)); pset.add((nx, ny))
        P[i, 0] = nx; P[i, 1] = ny
        ns = _score(P)
        if ns > cur_s or rng.random() < np.exp((ns - cur_s) / T):
            cur_s = ns
            if ns > best_s:
                best_s = ns
                best = P.copy()
        else:
            P[i, 0] = ox; P[i, 1] = oy
            pset.discard((nx, ny)); pset.add((ox, oy))
        T = max(0.05, T * 0.9995)
    return best, best_s

def _perturb(start, rng):
    start = start.copy()
    for _ in range(3):
        i = rng.randrange(_N)
        for _t in range(20):
            nx = max(_MINC, min(_MAXC, int(start[i, 0]) + rng.randint(-2, 2)))
            ny = max(_MINC, min(_MAXC, int(start[i, 1]) + rng.randint(-2, 2)))
            dup = False
            for p in range(_N):
                if p != i and start[p, 0] == nx and start[p, 1] == ny:
                    dup = True; break
            if not dup:
                start[i, 0] = nx; start[i, 1] = ny
                break
    return start

def construct():
    t_start = time.time()
    deadline = t_start + 80.0
    rng = random.Random(12345)

    seeds = [_seed_grid(), _seed_twoline(), _seed_random(rng)]
    best = None
    best_s = -1
    for s in seeds:
        sc = _score(s)
        if sc > best_s:
            best_s = sc
            best = s

    rounds = 6
    for r in range(rounds):
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - 5.0 - time.time()
        per = max(1.0, remaining / (rounds - r))
        start = _perturb(best, rng)
        cand, sc = _anneal(start, per, rng)
        if sc > best_s:
            best_s = sc
            best = cand

    return [tuple(int(v) for v in p) for p in best.tolist()]
