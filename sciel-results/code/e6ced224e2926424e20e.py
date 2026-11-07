"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _cycle_edges(cycle):
    m = len(cycle)
    out = []
    for i in range(m):
        out.append(_edge_key(cycle[i], cycle[(i + 1) % m]))
    return out


def _is_valid_cycle(cycle, adj, vertex_set):
    if not isinstance(cycle, (list, tuple)):
        return False
    m = len(cycle)
    if m < 3:
        return False
    if len(cycle) != len(set(cycle)):
        return False
    for v in cycle:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v not in vertex_set:
            return False
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        if b not in adj[a]:
            return False
    if set(cycle) != vertex_set:
        return False
    return True


def _is_valid_solution(cycles, adj, vertex_set):
    for c in cycles:
        if not _is_valid_cycle(c, adj, vertex_set):
            return False
    seen = set()
    for c in cycles:
        for e in _cycle_edges(c):
            if e in seen:
                return False
            seen.add(e)
    return True


def construct():
    n = P['n']
    adj = {u: set() for u in range(n)}
    edge_set = set()
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(_edge_key(u, v))

    vertex_set = set(range(n))

    # This graph is the circulant C(21, {±1, ±2, ±5}).
    # We can construct 3 edge-disjoint Hamiltonian cycles using step sizes 1, 2, 5.
    # Since gcd(k, 21) = 1 for k in {1, 2, 5}, each step generates a full Hamiltonian cycle.
    # The edge sets for steps 1, 2, 5 are disjoint since no step is ±1, ±2, or ±5 of another.

    def make_cycle(step):
        """Generate Hamiltonian cycle using constant step in circulant graph."""
        cycle = []
        v = 0
        for _ in range(n):
            cycle.append(v)
            v = (v + step) % n
        return cycle

    # Generate the three cycles
    c1 = make_cycle(1)
    c2 = make_cycle(2)
    c3 = make_cycle(5)

    # Validate all three
    cycles = []
    for c in [c1, c2, c3]:
        if _is_valid_cycle(c, adj, vertex_set):
            cycles.append(c)

    # Verify edge-disjointness
    if len(cycles) == 3 and _is_valid_solution(cycles, adj, vertex_set):
        return [[int(v) for v in c] for c in cycles]

    # Fallback: use DFS if circulant construction fails (should not happen)
    def _find_hamiltonian(adj, edge_set, used, deadline):
        n_local = P['n']
        nbrs = [sorted(adj[v]) for v in range(n_local)]
        start = 0
        adj_set = [set(nbrs[v]) for v in range(n_local)]

        def dfs():
            if time.time() > deadline:
                return None
            n_path = len(path)
            if n_path == n_local:
                last = path[-1]
                e = _edge_key(last, start)
                if e not in used and last in adj_set[start]:
                    return path[:]
                return None
            last = path[-1]
            unvisited_set = set(range(n_local)) - set(path)

            cands = []
            for y in nbrs[last]:
                if y == start:
                    continue
                if y in unvisited_set:
                    if _edge_key(last, y) not in used:
                        cands.append(y)

            if not cands:
                return None

            def fwd_count(y):
                k = 0
                for z in nbrs[y]:
                    if z == start:
                        continue
                    if z in unvisited_set:
                        if _edge_key(y, z) not in used:
                            k += 1
                return k

            cands.sort(key=fwd_count)

            for y in cands:
                path.append(y)
                res = dfs()
                if res is not None:
                    return res
                path.pop()
            return None

        path = [start]
        return dfs()

    deadline = time.time() + 70
    used = set()
    cycles = []
    for _ in range(3):
        c = _find_hamiltonian(adj, edge_set, used, deadline)
        if c is not None and _is_valid_cycle(c, adj, vertex_set):
            cycles.append(c)
            for e in _cycle_edges(c):
                used.add(e)

    while cycles and not _is_valid_solution(cycles, adj, vertex_set):
        cycles.pop()

    if not cycles:
        fb = list(range(n))
        return [[int(v) for v in fb]]

    return [[int(v) for v in c] for c in cycles]
# EVOLVE-BLOCK-END
