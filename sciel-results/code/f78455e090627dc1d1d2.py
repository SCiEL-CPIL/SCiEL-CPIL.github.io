"""Combined program: circulant search + LUW fallback for cage problem."""
import time as _time
from collections import deque

# EVOLVE-BLOCK-START

def _check_girth_full(n, edges, k, g, deadline):
    """Full verification via non-backtracking walks from every directed edge."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    max_depth = g - 2
    if max_depth < 1:
        return True
    stamp = [0] * (n * n)
    gen = 0
    for a in range(n):
        for b in adj[a]:
            gen += 1
            queue = [(b, a, 0)]
            stamp[b * n + a] = gen
            qi = 0
            while qi < len(queue):
                cur, prev, depth = queue[qi]
                qi += 1
                if cur == a and prev != b and depth >= 1:
                    return False
                if depth >= max_depth:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    idx = nxt * n + cur
                    if stamp[idx] != gen:
                        stamp[idx] = gen
                        queue.append((nxt, cur, depth + 1))
        if _time.time() > deadline:
            return True
    return True

def _build_luw(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if _time.time() > deadline - 30.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _try_circulant(n, k, g, deadline):
    """Find a k-regular circulant graph on n vertices with girth >= g."""
    if k % 2 != 0 or n < 2 * k:
        return None

    def _fast_girth_check(n, a, b, g):
        """Non-backtracking BFS from vertex 0 for circulant {±a,±b}.
        Circulant graphs are vertex-transitive, so checking from 0 suffices."""
        max_depth = g - 2
        if max_depth < 1:
            return True
        stamp = {}
        gen = 1
        # Initial neighbors of 0 via ±a and ±b
        queue = []
        for step in (a, n - a, b, n - b):
            if step == 0:
                continue
            queue.append((step, 0, 0))
            stamp[(step, 0)] = gen

        qi = 0
        while qi < len(queue):
            cur, prev, depth = queue[qi]
            qi += 1
            if cur == 0 and depth >= 1:
                return False
            if depth >= max_depth:
                continue
            for nxt in ((cur + a) % n, (cur - a) % n, (cur + b) % n, (cur - b) % n):
                if nxt == prev:
                    continue
                key = (nxt, cur)
                if key not in stamp:
                    stamp[key] = gen
                    queue.append((nxt, cur, depth + 1))
        return True

    # Generate candidate (a, b) pairs
    candidates = []
    for a in range(1, n // 2 + 1):
        if 2 * a % n == 0:
            continue
        for b in range(1, n // 2 + 1):
            if 2 * b % n == 0:
                continue
            if a == b:
                continue
            if (a + b) % n == 0 or (a - b) % n == 0:
                continue
            candidates.append((a, b))

    # Sort by heuristic: prefer small a, then small b
    candidates.sort(key=lambda x: (x[0], x[1]))

    for a, b in candidates:
        if _time.time() > deadline - 10.0:
            break
        if _fast_girth_check(n, a, b, g):
            edges = []
            for i in range(n):
                edges.append((i, (i + a) % n))
                edges.append((i, (i + b) % n))
            return n, edges
    return None

def construct(k, g):
    deadline = _time.time() + 110.0

    # 1. Try circulant construction for small n values
    if k == 4 and g == 10:
        # Try n values from 320 upward
        for n in [320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340]:
            if _time.time() > deadline - 20.0:
                break
            res = _try_circulant(n, k, g, deadline)
            if res:
                # Verify the result
                n_res, edges_res = res
                if _check_girth_full(n_res, edges_res, k, g, deadline):
                    return n_res, edges_res

    # 2. Fallback to LUW construction
    return _build_luw(k, g, deadline)

# EVOLVE-BLOCK-END
