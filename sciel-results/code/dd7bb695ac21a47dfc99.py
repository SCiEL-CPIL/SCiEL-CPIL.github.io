import time

_T0 = time.time()
_DEADLINE = _T0 + 140.0  # hard wall-clock safety margin


def _sq_dist(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return da * da + da * db + db * db


def _unit_neighbors(p):
    a, b = p
    return [(a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1)]


def _count_edges(pts):
    n = len(pts)
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _sq_dist(pts[i], pts[j]) == 1:
                cnt += 1
    return cnt


def _hex_ring(r):
    if r == 0:
        return [(0, 0)]
    # Walk the six edges of the hexagon of radius r.
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    p = [0, 0]
    for d in dirs:
        for _ in range(r):
            p[0] += d[0]
            p[1] += d[1]
            yield (p[0], p[1])


def _hex_spiral(n):
    pts = [(0, 0)]
    r = 1
    while len(pts) < n:
        for p in _hex_ring(r):
            if len(pts) < n:
                pts.append(p)
            else:
                break
        r += 1
    return pts


def _valid(pts):
    if len(pts) != 32:
        return False
    if len(set(pts)) != 32:
        return False
    for p in pts:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _sq_dist(pts[i], pts[j]) < 1:
                return False
    return True


def _polish(pts):
    pts = list(pts)
    best = _count_edges(pts)
    while time.time() < _DEADLINE - 1.0:
        improved = False
        for i in range(len(pts)):
            for cand in _unit_neighbors(pts[i]):
                if any(cand == pts[j] for j in range(len(pts)) if j != i):
                    continue
                if abs(cand[0]) > 64 or abs(cand[1]) > 64:
                    continue
                newpts = list(pts)
                newpts[i] = cand
                if not _valid(newpts):
                    continue
                e = _count_edges(newpts)
                if e > best:
                    pts = newpts
                    best = e
                    improved = True
        if not improved:
            break
    return pts, best


def construct():
    pts = _hex_spiral(32)
    if not _valid(pts):
        pts = [[a, b] for a in range(16) for b in range(2)]
    pts, _ = _polish(pts)
    if not _valid(pts):
        pts = [[a, b] for a in range(16) for b in range(2)]
    return [[a, b] for a, b in pts]
