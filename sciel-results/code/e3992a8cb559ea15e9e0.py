import time, random, math

# ---- precompute all nonconstant 4-term APs (0-based) ----
_APS = []
_AP_OF = [[] for _ in range(128)]
for _a in range(128):
    for _d in range(1, (128 - _a) // 3 + 1):
        _e = _a + 3 * _d
        if _e >= 128:
            break
        _idx = len(_APS)
        _APS.append((_a, _a + _d, _a + 2 * _d, _e))
        for _p in (_a, _a + _d, _a + 2 * _d, _e):
            _AP_OF[_p].append(_idx)

_TABLES = {}
def _table(k):
    if k not in _TABLES:
        t = [0] * (k ** 4)
        for a in range(k):
            for b in range(k):
                for c in range(k):
                    for d in range(k):
                        t[((a * k + b) * k + c) * k + d] = len((a, b, c, d))
        _TABLES[k] = t
    return _TABLES[k]

def _total_bad(c):
    bad = 0
    for t in _APS:
        a, b, cc, d = t
        if len((c[a], c[b], c[cc], c[d])) < 3:
            bad += 1
    return bad

def _sa(k, time_limit, rng):
    n = 128
    tbl = _table(k)
    A = _APS
    AOF = _AP_OF
    m = len(A)

    def init():
        return [rng.randrange(k) for _ in range(n)]

    def compute_bad(c):
        bad = [0] * m
        F = 0
        for i, t in enumerate(A):
            a, b, cc, d = t
            v = tbl[((c[a] * k + c[b]) * k + c[cc]) * k + c[d]]
            if v < 3:
                bad[i] = 1
                F += 1
        return bad, F

    start = time.time()
    patience_max = 4000
    while time.time() - start < time_limit:
        c = init()
        bad, F = compute_bad(c)
        if F == 0:
            return c
        bestF = F
        best = c[:]
        T = 1.0
        noimp = 0
        while time.time() - start < time_limit and noimp < patience_max:
            i = rng.randrange(n)
            old = c[i]
            new = rng.randrange(k - 1)
            if new >= old:
                new += 1
            deltas = []
            delta = 0
            for ai in AOF[i]:
                a, b, cc, d = A[ai]
                od = tbl[((c[a] * k + c[b]) * k + c[cc]) * k + c[d]]
                c[i] = new
                nd = tbl[((c[a] * k + c[b]) * k + c[cc]) * k + c[d]]
                c[i] = old
                ob = 1 if od < 3 else 0
                nb = 1 if nd < 3 else 0
                delta += nb - ob
                deltas.append((ai, nb))
            if delta <= 0 or (delta < 8 and rng.random() < math.exp(-delta / T)):
                c[i] = new
                for ai, nb in deltas:
                    F += nb - bad[ai]
                    bad[ai] = nb
                if F < bestF:
                    bestF = F
                    best = c[:]
                    noimp = 0
                    if F == 0:
                        return best
                else:
                    noimp += 1
                T *= 0.99985
            else:
                noimp += 1
        if bestF < F:
            c = best
            bad, F = compute_bad(c)
    return best if _total_bad(best) <= _total_bad(c) else c

def _fallback():
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3
                   for d in range(1, (x - 1) // 3 + 1)):
                colors.append(color)
                break
    return colors

def construct():
    rng = random.Random(20240517)
    deadline = time.time() + 80.0
    plan = [(3, 50.0), (4, 15.0), (5, 8.0), (6, 4.0), (7, 2.0)]
    for k, tlim in plan:
        remaining = deadline - time.time()
        if remaining < 1.5:
            break
        tlim = min(tlim, remaining - 1.0)
        if tlim <= 0:
            break
        res = _sa(k, tlim, rng)
        if res is not None and len(res) == 128 and _total_bad(res) == 0:
            return res
    return _fallback()
