import time
from random import Random

_T0 = time.time()
_DEADLINE = 145.0


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def _build_instance(n, d, seed):
    rng = Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edges:
            continue
        if degree[u] < d and degree[v] < d:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _build_conflict_bitsets(edges, n):
    m = len(edges)
    edge_set = set(edges)
    conflict = [0] * m
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
            else:
                if c in adj[a] or c in adj[b] or d in adj[a] or d in adj[b]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
    return conflict


def _greedy_dsatur(conflict):
    m = len(conflict)
    if m == 0:
        return [], 0
    colors = [-1] * m
    sat = [set() for _ in range(m)]
    uncolored = list(range(m))
    color_count = 0

    while uncolored:
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            s = len(sat[v])
            dg = conflict[v].bit_count()
            if s > best_sat or (s == best_sat and dg > best_deg):
                best_v = v
                best_sat = s
                best_deg = dg

        c = 0
        while c in sat[best_v]:
            c += 1
        colors[best_v] = c
        if c == color_count:
            color_count += 1

        for nb in range(m):
            if colors[nb] == -1 and (conflict[best_v] >> nb) & 1:
                sat[nb].add(c)

        uncolored.remove(best_v)

    return colors, color_count


def _max_clique_lower_bound(conflict):
    m = len(conflict)
    if m == 0:
        return 0
    nodes = list(range(m))
    nodes.sort(key=lambda x: conflict[x].bit_count(), reverse=True)
    best_clique_size = 1
    for start in nodes[:10]:
        clique = [start]
        candidates = [v for v in nodes if v > start and (conflict[start] >> v) & 1]
        while candidates:
            best_c = -1
            for v in candidates:
                if all((conflict[v] >> u) & 1 for u in clique):
                    if best_c == -1 or conflict[v].bit_count() > conflict[best_c].bit_count():
                        best_c = v
            if best_c == -1:
                break
            clique.append(best_c)
            candidates = [v for v in candidates if v != best_c and (conflict[best_c] >> v) & 1]
        if len(clique) > best_clique_size:
            best_clique_size = len(clique)
    return best_clique_size


def _backtracking_dsatur(conflict, K, time_limit):
    m = len(conflict)
    if m == 0:
        return [0] * 0
    if K == 0:
        return None
    colors = [-1] * m
    sat = [set() for _ in range(m)]
    uncolored = set(range(m))
    degrees = [conflict[i].bit_count() for i in range(m)]
    start_time = time.time()

    def solve():
        if not uncolored:
            return True
        if time.time() - start_time > time_limit:
            return False
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in uncolored:
            s = len(sat[v])
            d = degrees[v]
            if s > best_sat or (s == best_sat and d > best_deg):
                best_v = v
                best_sat = s
                best_deg = d
        if best_v == -1:
            return False
        used_colors = sat[best_v]
        for c in range(K):
            if c not in used_colors:
                colors[best_v] = c
                uncolored.remove(best_v)
                neighbors_to_update = []
                nb_mask = conflict[best_v]
                for nb in uncolored:
                    if (nb_mask >> nb) & 1:
                        neighbors_to_update.append(nb)
                for nb in neighbors_to_update:
                    sat[nb].add(c)
                if solve():
                    return True
                for nb in neighbors_to_update:
                    sat[nb].discard(c)
                uncolored.add(best_v)
                colors[best_v] = -1
        return False

    if solve():
        return colors
    return None


def _solve_instance(n, d, seed):
    edges = _build_instance(n, d, seed)
    m = len(edges)
    if m == 0:
        return []
    conflict = _build_conflict_bitsets(edges, n)
    greedy_colors, ub = _greedy_dsatur(conflict)
    lb = _max_clique_lower_bound(conflict)
    best_colors = greedy_colors
    best_k = ub
    remaining_time = 40.0
    for k in range(lb, ub):
        if remaining_time < 2.0:
            break
        t_lim = min(5.0, remaining_time)
        start_t = time.time()
        colors = _backtracking_dsatur(conflict, k, t_lim)
        elapsed = time.time() - start_t
        remaining_time -= elapsed
        if colors is not None:
            best_colors = colors
            best_k = k
            break
    return best_colors


def construct():
    global _T0
    _T0 = time.time()
    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if _time_left() < 15.0:
            edges = _build_instance(*spec)
            n = spec[0]
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            conflict = _build_conflict_bitsets(edges, n)
            colors, _ = _greedy_dsatur(conflict)
            results.append(colors)
            continue
        colors = _solve_instance(*spec)
        results.append(colors)
    return results
