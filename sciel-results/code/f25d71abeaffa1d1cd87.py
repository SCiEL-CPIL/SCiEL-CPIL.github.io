import time
import math

N = 12
START = time.time()
DEADLINE = START + 130.0
EPS = 1e-9

# ---- hexagonal lattice basis (spacing 1.0 => unit hexagons touch) ----
a1 = (math.sqrt(3.0) / 2.0, 0.5)
a2 = (math.sqrt(3.0) / 2.0, -0.5)

# ---- precompute lattice sites & adjacency ----
def _build():
    sites = []
    seen = set()
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = (i * a1[0] + j * a2[0], i * a1[1] + j * a2[1])
            key = (round(p[0], 6), round(p[1], 6))
            if key in seen:
                continue
            seen.add(key)
            sites.append(p)
    pos = sites
    n = len(pos)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
            if d < 1.0 + 1e-6:
                adj[i].add(j)
                adj[j].add(i)
    return pos, adj


_POS, _ADJ = _build()


def dist_to_hex_boundary(q, S):
    x = abs(q[0]); y = abs(q[1])
    h = math.sqrt(3.0) / 2.0
    if x > h * S:
        return x - h * S
    if y > S:
        return y - S
    if y > h * (S - x / math.sqrt(3.0)):
        return y - h * (S - x / math.sqrt(3.0))
    return 0.0


def min_side(points, ang):
    best = 0.0
    for p in points:
        c, s = math.cos(ang), math.sin(ang)
        lx = c * p[0] + s * p[1]
        ly = -s * p[0] + c * p[1]
        v = dist_to_hex_boundary((lx, ly), 1.0)
        if v > best:
            best = v
    return best


def cluster_side(coords, ang):
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return min_side([(p[0] - cx, p[1] - cy) for p in coords], ang)


def optimize_cluster(coords):
    lo, hi = -math.pi / 6.0, math.pi / 6.0
    gr = (math.sqrt(5.0) + 1.0) / 2.0
    a = hi - (hi - lo) / gr
    b = lo + (hi - lo) / gr
    fa = cluster_side(coords, a); fb = cluster_side(coords, b)
    for _ in range(26):
        if time.time() > DEADLINE:
            break
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - (hi - lo) / gr
            fa = cluster_side(coords, a)
        else:
            lo, a, fa = a, b, fb
            b = lo + (hi - lo) / gr
            fb = cluster_side(coords, b)
    ang = (lo + hi) / 2.0
    S = cluster_side(coords, ang)
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return S, ang, cx, cy


def build_clusters():
    pos = _POS; adj = _ADJ; n = len(pos)
    clusters = set()
    for start in range(n):
        def grow(free, cur):
            if len(cur) == N:
                clusters.add(frozenset(cur)); return
            if time.time() > DEADLINE:
                return
            for i in sorted(free):
                if i in cur:
                    continue
                if any(j in cur for j in adj[i]):
                    cur.add(i); grow(free, cur); cur.discard(i)
        grow(set(range(n)), {start})
    return [list(c) for c in clusters]


def construct():
    global START, DEADLINE
    START = time.time()
    DEADLINE = START + 130.0

    clusters = build_clusters()
    best = None
    for c in clusters:
        if time.time() > DEADLINE:
            break
        coords = [_POS[k] for k in c]
        S, ang, cx, cy = optimize_cluster(coords)
        if best is None or S < best[0] - 1e-9:
            best = (S, ang, cx, cy, c)

    S, ang, cx, cy, c = best
    inner = []
    for k in c:
        p = _POS[k]
        local = (p[0] - cx, p[1] - cy)
        c2, s2 = math.cos(ang), math.sin(ang)
        wx = c2 * local[0] - s2 * local[1]
        wy = s2 * local[0] + c2 * local[1]
        inner.append([wx, wy, math.degrees(ang)])

    return inner, (0.0, 0.0), S, math.degrees(ang)
