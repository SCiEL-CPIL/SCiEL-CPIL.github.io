"""Triangle-free diameter-four augmentation - structural combinatorial approach."""
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = set()
for e in P['edges']:
    u, v = e
    BASE.add((min(u, v), max(u, v)))

BASE_ADJ = [set() for _ in range(N)]
for (u, v) in BASE:
    BASE_ADJ[u].add(v)
    BASE_ADJ[v].add(u)


def _check(added):
    """Return True if added edges make graph connected, triangle-free, diameter<=4."""
    adj = [set(BASE_ADJ[i]) for i in range(N)]
    for (u, v) in added:
        adj[u].add(v)
        adj[v].add(u)
    # Triangle-free check
    for u in range(N):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    # Connected + diameter via BFS from each vertex
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        for v in range(N):
            if dist[v] == -1 or dist[v] > 4:
                return False
    return True


def _gen_star(center, far_set):
    """Generate star edges from center to far_set (vertices at path-dist >= 2 from center)."""
    edges = []
    for v in far_set:
        if v != center:
            u, w = (center, v) if center < v else (v, center)
            if (u, w) not in BASE:
                edges.append((u, w))
    return edges


def _gen_star2(c1, c2, far1, far2):
    """Two-star: edges from c1 to far1, from c2 to far2."""
    edges = []
    for v in far1:
        if v != c1:
            u, w = (c1, v) if c1 < v else (v, c1)
            if (u, w) not in BASE:
                edges.append((u, w))
    for v in far2:
        if v != c2:
            u, w = (c2, v) if c2 < v else (v, c2)
            if (u, w) not in BASE:
                edges.append((u, w))
    return edges


def _far_from(c, min_dist=2):
    """Vertices at path-distance >= min_dist from c."""
    return [v for v in range(N) if abs(v - c) >= min_dist]


def _try_solution(added):
    """Verify a candidate solution."""
    # Check no duplicates, no base edges
    seen = set()
    for (u, v) in added:
        key = (min(u, v), max(u, v))
        if key in BASE or key in seen:
            return False
        seen.add(key)
    return _check(added)


def solve():
    deadline = time.time() + 80
    best = None
    best_count = float('inf')

    def update(added):
        nonlocal best, best_count
        if len(added) < best_count and _try_solution(added):
            best = list(added)
            best_count = len(added)

    # Strategy 1: Single star from center c to all far vertices (dist >= 2)
    for c in range(N):
        if time.time() > deadline:
            break
        far = _far_from(c, 2)
        edges = _gen_star(c, far)
        update(edges)

    # Strategy 2: Single star to subset of far vertices (dist >= 3, 4, 5)
    for c in range(N):
        if time.time() > deadline:
            break
        for min_d in [3, 4, 5]:
            far = _far_from(c, min_d)
            edges = _gen_star(c, far)
            update(edges)
            # Also try adding a few more
            # Try star to dist>=2 but skip some
            far2 = _far_from(c, 2)
            for skip in range(len(far2)):
                if time.time() > deadline:
                    break
                subset = far2[:skip] + far2[skip+1:]
                edges = _gen_star(c, subset)
                update(edges)

    # Strategy 3: Two centers
    for c1 in range(N):
        for c2 in range(c1 + 1, N):
            if time.time() > deadline:
                break
            for min_d1 in [3, 4]:
                for min_d2 in [3, 4]:
                    far1 = _far_from(c1, min_d1)
                    far2 = _far_from(c2, min_d2)
                    edges = _gen_star2(c1, c2, far1, far2)
                    update(edges)

    # Strategy 4: Center + a few extra edges to cover remaining
    for c in range(N):
        if time.time() > deadline:
            break
        base_edges = _gen_star(c, _far_from(c, 2))
        # Try removing edges and adding different ones
        for i in range(len(base_edges)):
            if time.time() > deadline:
                break
            trial = base_edges[:i] + base_edges[i+1:]
            update(trial)
            # Try adding an edge not in star
            removed = base_edges[i]
            for v in range(N):
                if time.time() > deadline:
                    break
                u, w = (c, v) if c < v else (v, c)
                if (u, w) in BASE or (u, w) == removed:
                    continue
                if (u, w) in [tuple(sorted(e)) for e in trial]:
                    continue
                trial2 = trial + [(u, w)]
                update(trial2)

    # Strategy 5: Direct enumeration of small edge sets (up to 5 edges)
    # Use structured candidates: edges that skip at least 2 path positions
    candidates = []
    for u in range(N):
        for v in range(u + 3, N):  # skip at least 2 to avoid creating triangles with path
            if (u, v) not in BASE:
                candidates.append((u, v))

    # Greedy: pick edges that reduce diameter most
    if time.time() < deadline:
        added = []
        added_set = set()
        for _ in range(10):
            if time.time() > deadline:
                break
            # Find the pair with max distance in current graph
            adj = [set(BASE_ADJ[i]) for i in range(N)]
            for (u, v) in added:
                adj[u].add(v)
                adj[v].add(u)
            # BFS from 0 and N-1 to find farthest
            max_dist = 0
            worst_pair = None
            for s in [0, N - 1, N // 2]:
                dist = [-1] * N
                dist[s] = 0
                dq = deque([s])
                while dq:
                    u = dq.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            dq.append(v)
                for v in range(N):
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                        worst_pair = (s, v)
            if max_dist <= 4:
                break
            # Try all candidate edges, pick best
            best_edge = None
            best_new_max = max_dist + 1
            for e in candidates:
                if e in added_set:
                    continue
                trial = added + [e]
                if not _try_solution(trial):
                    continue
                # Compute diameter
                adj2 = [set(BASE_ADJ[i]) for i in range(N)]
                for (u, v) in trial:
                    adj2[u].add(v)
                    adj2[v].add(u)
                diam = 0
                for s in range(N):
                    dist = [-1] * N
                    dist[s] = 0
                    dq = deque([s])
                    while dq:
                        u = dq.popleft()
                        for v in adj2[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                dq.append(v)
                    for v in range(N):
                        if dist[v] > diam:
                            diam = dist[v]
                if diam < best_new_max:
                    best_new_max = diam
                    best_edge = e
                    if diam <= 4:
                        break
            if best_edge is None:
                break
            added.append(best_edge)
            added_set.add(best_edge)
        if max_dist <= 4 or _try_solution(added):
            update(added)

    # Final: try to reduce best solution
    if best is not None:
        cur = list(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(cur)):
                if time.time() > deadline:
                    break
                trial = cur[:i] + cur[i+1:]
                if _try_solution(trial):
                    cur = trial
                    improved = True
                    break
        if len(cur) < best_count:
            best = cur
            best_count = len(cur)

    if best is None:
        return []
    return [[int(u), int(v)] for (u, v) in best]


def construct():
    result = solve()
    return result
