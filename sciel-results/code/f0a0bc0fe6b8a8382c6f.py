"""Decompose fixed instance graphs into few cycles + edges (Erdős-Gallai style)."""
import time
from collections import defaultdict, deque
from random import Random


def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, n, time_limit=40.0):
    """
    Decompose a graph into edge-disjoint cycles and edges, minimizing total pieces.
    Strategy:
      1. Repeatedly find a long cycle (using DFS with backtracking, trying to maximize
         cycle length) and extract it.
      2. After no more cycles can be found, remaining edges become single-edge pieces.
      3. Use a time budget to ensure we finish in time.
    """
    if not edges:
        return []

    # Build adjacency
    adj = defaultdict(set)
    alive_edges = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        alive_edges.add((u, v))
        alive_edges.add((v, u))

    pieces = []
    start_time = time.time()

    def try_find_long_cycle(start_edge, min_cycle_len=3, max_search=50000):
        """
        Try to find a long cycle containing start_edge (a,b).
        Uses DFS from a to b avoiding the direct edge (a,b), trying to find
        a path of length >= min_cycle_len - 1 (i.e., at least min_cycle_len - 1 edges
        in the path, so the cycle has at least min_cycle_len vertices).
        Returns the cycle as a list of vertices, or None.
        """
        a, b = start_edge
        # We need a path from a to b that doesn't use edge (a,b) directly
        # and has at least 2 edges (so cycle has at least 3 vertices)
        found = [None]
        node_count = [0]

        def dfs(u, path, visited):
            if found[0] is not None or node_count[0] > max_search:
                return
            node_count[0] += 1
            if u == b and len(path) >= 3:
                found[0] = list(path)
                return
            for v in sorted(adj[u]):
                if v in visited:
                    continue
                if (u, v) == (a, b) or (u, v) == (b, a):
                    continue
                # Check edge is alive
                if (u, v) not in alive_edges:
                    continue
                visited.add(v)
                path.append(v)
                dfs(v, path, visited)
                if found[0] is not None:
                    return
                path.pop()
                visited.discard(v)

        visited = {a}
        dfs(a, [a], visited)
        return found[0]

    def find_best_cycle():
        """
        Find the longest cycle in the remaining graph.
        Try each edge as a potential start, and find the longest cycle through it.
        For efficiency, we sample edges and try to find good cycles.
        """
        if time.time() - start_time > time_limit:
            return None, None

        # Get unique edges
        unique_edges = set()
        for u, v in alive_edges:
            if u < v:
                unique_edges.add((u, v))

        if not unique_edges:
            return None, None

        best_cycle = None
        best_len = 0

        # Try to find a long cycle by trying several edges
        # Prioritize edges in high-degree areas
        edge_list = sorted(unique_edges, key=lambda e: (len(adj[e[0]]) + len(adj[e[1]]), e), reverse=True)

        # Limit the number of edges we try to stay within time
        max_edges_to_try = 100
        edges_to_try = edge_list[:max_edges_to_try]

        for edge in edges_to_try:
            if time.time() - start_time > time_limit:
                break
            if edge not in unique_edges:
                continue
            cycle = try_find_long_cycle(edge)
            if cycle is not None:
                clen = len(cycle)
                if clen > best_len:
                    best_len = clen
                    best_cycle = cycle

        return best_cycle, best_len

    # Main loop: extract cycles
    while True:
        if time.time() - start_time > time_limit:
            break
        if not alive_edges:
            break

        cycle, clen = find_best_cycle()
        if cycle is None:
            break

        # Remove edges in the cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            alive_edges.discard((u, v))
            alive_edges.discard((v, u))
            adj[u].discard(v)
            adj[v].discard(u)

        pieces.append([int(v) for v in cycle])

    # Remaining edges
    remaining = set()
    for u, v in alive_edges:
        if u < v:
            remaining.add((u, v))

    for u, v in sorted(remaining):
        pieces.append([u, v])

    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    start_time = time.time()
    results = []

    for i, (n, p, seed) in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = 145.0 - elapsed
        graphs_left = len(specs) - i
        alloc_time = remaining_time / graphs_left
        alloc_time = max(2.0, min(alloc_time, 45.0))

        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=alloc_time)
        results.append(pieces)

    return results
