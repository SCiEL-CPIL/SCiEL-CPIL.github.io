"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 6.0


def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False

    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Girth check via non-backtracking BFS from each vertex.
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
    return True


def _check_girth_fast(adj, n, g):
    """
    Fast girth check. Returns True if girth >= g.
    """
    for s in range(n):
        dq = deque()
        visited = {}
        for nb in adj[s]:
            dq.append((nb, s, 1))
            visited[(nb, s)] = 1
        while dq:
            curr, prev, depth = dq.popleft()
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
            if depth + 1 >= g:
                continue
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in visited:
                    visited[state] = depth + 1
                    dq.append((nb, curr, depth + 1))
    return True


def _circulant_graph(n, a, b):
    """
    Generate edges for circulant graph C_n(a, b).
    """
    edges = set()
    for i in range(n):
        for j in [(i + a) % n, (i - a) % n, (i + b) % n, (i - b) % n]:
            if i < j:
                edges.add((i, j))
            elif i > j:
                edges.add((j, i))
    return list(edges)


def _find_circulant_cage(k, g, max_n=320):
    """
    Search for a k-regular circulant graph with girth >= g and n <= max_n.
    """
    # For k=4, generators are +/-a, +/-b
    # We assume a, b are such that the graph is simple and connected.
    for n in range(2 * k + 1, max_n + 1):
        # a, b must be coprime to n for connectivity? Not strictly, but good.
        # Also a != b, a != -b, a != 0, b != 0.
        for a in range(1, n // 2 + 1):
            for b in range(a + 1, n // 2 + 1):
                if a == b or a + b == n:
                    continue
                edges = _circulant_graph(n, a, b)
                if len(edges) != n * k // 2:
                    continue
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                if _check_girth_fast(adj, n, g):
                    return n, edges
    return None, None


def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges).
    """
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if (g - 5) % 2 == 0:
        d = g - 4

    if q == 4:
        table = [[0, 0, 0, 0],
                 [0, 1, 2, 3],
                 [0, 2, 3, 1],
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges


class _ShortCycleChecker:
    """
    Fast checker for non-backtracking walks that close into a cycle of length
    < g. Returns True as soon as any such cycle is found (early termination).
    """

    def __init__(self, adj, n, g):
        self.adj = adj
        self.n = n
        self.g = g
        self.max_depth = g - 2
        self.visited = [0] * n
        self.round = 0
        self.dq = deque()

    def has_short_cycle(self):
        adj = self.adj
        n = self.n
        g = self.g
        visited = self.visited
        max_depth = self.max_depth
        dq = self.dq

        for s in range(n):
            self.round += 1
            rnd = self.round
            dq.clear()
            for nb in adj[s]:
                if visited[nb] != rnd:
                    visited[nb] = rnd
                    dq.append((nb, s, 1))

            while dq:
                curr, prev, depth = dq.popleft()
                if curr == s:
                    if depth + 1 < g:
                        return True
                    continue
                if depth >= max_depth:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if visited[nb] != rnd:
                        visited[nb] = rnd
                        dq.append((nb, curr, depth + 1))

        return False

    def count(self):
        adj = self.adj
        n = self.n
        g = self.g
        visited = self.visited
        max_depth = self.max_depth
        count = 0
        dq = self.dq

        for s in range(n):
            self.round += 1
            rnd = self.round
            dq.clear()
            for nb in adj[s]:
                if visited[nb] != rnd:
                    visited[nb] = rnd
                    dq.append((nb, s, 1))

            while dq:
                curr, prev, depth = dq.popleft()
                if curr == s:
                    if depth + 1 < g:
                        count += 1
                    continue
                if depth >= max_depth:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if visited[nb] != rnd:
                        visited[nb] = rnd
                        dq.append((nb, curr, depth + 1))

        return count


def _count_short_cycles(adj, n, g):
    """Count cycles of length < g using non-backtracking walks."""
    return _ShortCycleChecker(adj, n, g).count()


def _has_short_cycle(adj, n, g):
    """Check if any cycle of length < g exists (early termination)."""
    return _ShortCycleChecker(adj, n, g).has_short_cycle()


def _generate_4_regular(n, seed):
    """
    Generate a random 4-regular simple graph on n vertices using configuration
    model with rejection and retry.
    """
    rng = random.Random(seed)
    for _ in range(80):
        stubs = []
        for v in range(n):
            stubs.extend((v, v, v, v))
        rng.shuffle(stubs)

        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (u, v) if u < v else (v, u)
            if edge in edges:
                valid = False
                break
            edges.add(edge)

        if not valid:
            continue

        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == 4 for d in deg):
            return list(edges)

    return None


def _improve_girth(n, edges, g, time_limit, rng):
    """
    Simulated annealing with 2-switch operations to increase girth.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v) if u < v else (v, u))

    checker = _ShortCycleChecker(adj, n, g)
    current_has_cycle = checker.has_short_cycle()
    if not current_has_cycle:
        return list(edges), 0

    current_cost = checker.count()
    best_edges = list(edges)
    best_cost = current_cost

    t_start = time.time()
    temp = 50.0
    cooling = 0.999
    min_temp = 0.1
    inv_k = 1.0 / 50.0

    while time.time() < t_start + time_limit and temp > min_temp:
        e1 = rng.choice(edges)
        e2 = rng.choice(edges)
        if e1 == e2:
            continue

        u1, v1 = e1
        u2, v2 = e2

        if len({u1, v1, u2, v2}) < 4:
            continue

        if rng.random() < 0.5:
            na, nb = (u1, u2)
            nc, nd = (v1, v2)
        else:
            na, nb = (u1, v2)
            nc, nd = (v1, u2)

        new_e1 = (na, nb) if na < nb else (nb, na)
        new_e2 = (nc, nd) if nc < nd else (nd, nc)

        if new_e1 == new_e2:
            continue
        if new_e1 in edge_set or new_e2 in edge_set:
            continue

        edge_set.discard(e1)
        edge_set.discard(e2)
        edge_set.add(new_e1)
        edge_set.add(new_e2)

        adj[u1].discard(v1); adj[v1].discard(u1)
        adj[u2].discard(v2); adj[v2].discard(u2)
        adj[na].add(nb); adj[nb].add(na)
        adj[nc].add(nd); adj[nd].add(nc)

        edges.remove(e1)
        edges.remove(e2)
        edges.append(new_e1)
        edges.append(new_e2)

        new_has_cycle = checker.has_short_cycle()
        if not new_has_cycle:
            new_cost = 0
        else:
            new_cost = checker.count()

        if new_cost < best_cost:
            best_cost = new_cost
            best_edges = list(edges)

        if new_cost <= current_cost or rng.random() < pow(2.718, -(new_cost - current_cost) * inv_k / temp):
            current_cost = new_cost
            current_has_cycle = new_has_cycle
        else:
            edge_set.discard(new_e1)
            edge_set.discard(new_e2)
            edge_set.add(e1)
            edge_set.add(e2)

            adj[na].discard(nb); adj[nb].discard(na)
            adj[nc].discard(nd); adj[nd].discard(nc)
            adj[u1].add(v1); adj[v1].add(u1)
            adj[u2].add(v2); adj[v2].add(u2)

            edges.remove(new_e1)
            edges.remove(new_e2)
            edges.append(e1)
            edges.append(e2)

        temp *= cooling

    return best_edges, best_cost


def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN

    # 1. Try to find a Circulant Graph solution.
    if k == 4 and g == 10:
        n_circ, edges_circ = _find_circulant_cage(4, 10, max_n=320)
        if n_circ is not None:
            if _validate_graph(n_circ, edges_circ, k, g):
                return n_circ, edges_circ

    # 2. LUW baseline.
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # 3. Random/SA search for smaller graphs.
    if k == 4 and g == 10:
        rng = random.Random(42)
        candidates = [320, 340, 360, 400]
        candidates = [s for s in candidates if s < best_n]

        for size in candidates:
            if time.time() > deadline:
                break
            time_left = deadline - time.time()
            if time_left < 3:
                break

            for attempt in range(5):
                if time.time() > deadline:
                    break
                edges = _generate_4_regular(size, seed=1000 + size * 31 + attempt * 7)
                if edges is None:
                    continue

                adj = [set() for _ in range(size)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)

                checker = _ShortCycleChecker(adj, size, g)
                if not checker.has_short_cycle():
                    if _validate_graph(size, edges, k, g):
                        return size, edges

                time_left = deadline - time.time()
                if time_left < 4:
                    break

                sa_time = max(2.0, (time_left - 2.0) * 0.15)
                improved_edges, cost = _improve_girth(size, edges, g, sa_time, rng)
                if cost == 0:
                    if _validate_graph(size, improved_edges, k, g):
                        return size, improved_edges

    return best_n, best_edges

# EVOLVE-BLOCK-END
