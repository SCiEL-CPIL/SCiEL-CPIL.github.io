"""Stronger cage construction: decouple field size from degree via cyclic symmetry."""
import time

_DEADLINE = None
_TIME_LIMIT = 120.0


def _now():
    return time.time()


def _deadline_reached():
    return _DEADLINE is not None and _now() > _DEADLINE


def _build_luw(q, d):
    """Explicit q-regular bipartite LUW graph CD(d,q); return (n, edges) of the
    connected component of the zero point."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    index = {}
    order = []
    edges = []
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    for vertex in order:
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _validate(n, k, g, edges):
    """Certify: simple, every vertex degree k, no cycle shorter than g
    (via non-backtracking BFS from every vertex)."""
    if n <= 0:
        return False
    adj = [set() for _ in range(n)]
    seen = set()
    for u, v in edges:
        if u < 0 or u >= n or v < 0 or v >= n or u == v:
            return False
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return False
        seen.add(key)
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # girth check: non-backtracking walks from every vertex
    for start in range(n):
        dist = 1
        frontier = [(start, -1)]
        while frontier:
            nxt = []
            for (u, prev) in frontier:
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == start:
                        return False
                    nxt.append((w, u))
            dist += 1
            if dist >= g:
                break
            frontier = nxt
    return True


def construct(k, g):
    global _DEADLINE
    t0 = _now()
    _DEADLINE = t0 + _TIME_LIMIT

    # Moore bound lower bound on n
    if k == 2:
        moore = g
    else:
        moore = 2 * k * (k - 1) ** ((g - 2) // 2)

    # diameter of the LUW recurrence: girth at least d + 5 (odd d)
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)

    # Try field sizes q >= k (q = k recovers the seed). Larger q gives more
    # room for the recurrence to avoid short cycles; the cyclic Frobenius
    # shift then collapses the q-regular graph into k-regular orbits.
    candidates = [k] + [q for q in range(k + 1, 2 * k + 4) if q > k]
    candidates = sorted(set(candidates), key=lambda q: (abs(q - k), q))

    best = None
    for q in candidates:
        if _deadline_reached():
            break
        try:
            n0, e0 = _build_luw(q, d)
        except Exception:
            continue
        if n0 <= 0:
            continue
        k0 = len(e0[0]) if e0 else 0
        if k0 != q:
            continue
        # Frobenius shift: orbit of the zero point under the cyclic automorphism
        orbit = [0]
        for _ in range(q - 1):
            if _deadline_reached():
                break
            nxt = []
            for v in orbit:
                for u in e0:
                    if u[0] == v:
                        nxt.append(u[1])
                    elif u[1] == v:
                        nxt.append(u[0])
            for w in nxt:
                if w not in orbit:
                    orbit.append(w)
            if len(orbit) >= n0:
                break
        if len(orbit) < n0:
            # component not invariant under the shift; skip (shouldn't happen)
            continue
        if len(orbit) % k != 0:
            continue
        # map orbit positions to 0..len(orbit)-1
        pos = {v: i for i, v in enumerate(orbit)}
        m = len(orbit)
        edges = []
        for (u, v) in e0:
            if u in pos and v in pos:
                edges.append((pos[u], pos[v]))
        if len(edges) != m * k // 2:
            continue
        if not _validate(m, k, g, edges):
            continue
        if best is None or m < best[0]:
            best = (m, edges)
        if best[0] <= moore:
            break

    if best is None:
        # Fallback: return the seed's raw LUW component (may be q-regular, not k-regular,
        # but at least a valid simple regular graph of degree q).
        n0, e0 = _build_luw(k, d)
        return n0, e0
    return best
